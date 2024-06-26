import os
import torch
import time
import numpy as np
import pickle

from tensorboardX import SummaryWriter
from torch import nn
from logging import getLogger
from data import Vocab, NLP, S2SDataset
from utils import build_optimizer, init_seed, init_logger, init_device, read_configuration, collate_fn_graph_text, \
    format_time
from module import GraphEncoder, GraphReconstructor, GraphPointer, Memory_Shift, GenProjection, StaticPlanReconstruction
from transformers import BartTokenizer, BartForConditionalGeneration, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader


def compute_kd_loss(node_embeddings, desc_embeddings, node_masks, kd_masks):
    assert node_embeddings.size() == desc_embeddings.size()
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(node_embeddings, desc_embeddings)
    loss = loss.mean(dim=-1)
    masks = node_masks * kd_masks
    loss = loss.masked_select(masks).mean()
    return loss


def compute_ce_loss(logits, labels, masks):
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.reshape_as(labels)
    loss = loss.masked_select(masks).mean()
    return loss


def compute_gen_loss(logits, labels, masks):
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.reshape_as(labels)
    loss = loss.masked_select(masks).mean()
    return loss


def compute_gen_acc(logits, labels, masks):
    gentext_idx = torch.max(logits, dim=-1)[1]
    eqne = torch.eq(gentext_idx, labels)
    acc = torch.sum(eqne.masked_select(masks)) / torch.sum(masks)
    return acc


def computer_coverage_loss(coverage_pro, nodes):
    mse_loss = nn.MSELoss(reduction='none')
    label = torch.zeros(coverage_pro.size(0), coverage_pro.size(1)).cuda()
    node_mask = label.scatter_(1, nodes, torch.ones_like(label).cuda())
    index = torch.LongTensor([0]).cuda()
    node_mask.index_fill_(1, index, 0)
    loss = mse_loss(coverage_pro, node_mask)
    loss = loss.mean()
    return loss


def computer_copy_loss(copy, pointers, pointers_mask):
    # mse_loss = nn.MSELoss(reduction='none')
    # loss = mse_loss(copy, pointers)
    # loss = torch.masked_select(loss, pointers_mask).mean()
    copy_prob = torch.where(pointers.bool(), 1 - copy, copy)
    loss = copy_prob.masked_select(pointers_mask).mean()
    return loss


def computer_sprec_loss(sp_pro, nodes_idx, node_mask):
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = ce_loss(sp_pro.view(-1, sp_pro.size(-1)), nodes_idx.view(-1))
    loss = loss.reshape_as(nodes_idx)
    loss = loss.masked_select(node_mask).mean()
    return loss


def run_train_batch(config, batch, student, plm, sprec, reconstructor, copyer, memory, genproject,
                    plm_optimizer, external_optimizer, device, nodeidx2bartidx):
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
    recon_relations, relations_masks, recon_positions, recon_masks, gen_outputs, gen_masks, \
    pointer, pointer_masks, heads, tails, tri_mask = batch

    kd_description = kd_description.to(device)
    kd_description_masks = kd_description_masks.to(device)
    # output_dict = teacher(input_ids=kd_description,
    #                       attention_mask=kd_description_masks,
    #                       output_hidden_states=True,
    #                       return_dict=True)
    # positions = kd_positions.unsqueeze(-1).expand(-1, -1, output_dict["encoder_last_hidden_state"].size(-1)).to(device)
    # teacher_embeddings = torch.gather(output_dict["encoder_last_hidden_state"], dim=1, index=positions)
    # teacher_embeddings = teacher_embeddings.detach()

    nodes = nodes.to(device)
    student_embeddings, vocabnode_embedding = student(nodes, edges, types)

    node_masks = node_masks.to(device)
    kd_masks = torch.ne(kd_positions, 0).to(device)
    # kd_loss = compute_kd_loss(student_embeddings, teacher_embeddings, node_masks, kd_masks)

    gen_outputs = gen_outputs.to(device)
    gen_masks = gen_masks.to(device)
    output_dict = plm(input_ids=None,
                      inputs_embeds=student_embeddings,
                      attention_mask=node_masks,
                      decoder_input_ids=gen_outputs[:, :-1],
                      decoder_attention_mask=gen_masks[:, :-1],
                      output_hidden_states=True,
                      labels=gen_outputs[:, 1:].contiguous(),
                      return_dict=True)
    plm_loss = output_dict["loss"]
    hidden_state = output_dict["decoder_hidden_states"]

    tri_mask = tri_mask.to(device)
    recon_relations = recon_relations.to(device)
    recon_masks = recon_masks.to(device)
    qt = memory(hidden_state, heads, tails, tri_mask, recon_relations, student_embeddings)

    decoder_input_embeddings = plm.get_input_embeddings()(gen_outputs[:, :-1])
    decoder_output_hiddens = output_dict["decoder_hidden_states"][-1]

    total_pro, coverage, pro_switch = genproject(decoder_output_hiddens, qt, decoder_input_embeddings, edges,
                                                 nodeidx2bartidx)
    gen_loss = compute_gen_loss(total_pro, gen_outputs[:, 1:].contiguous(), gen_masks[:, :-1])
    coverage_loss = computer_coverage_loss(coverage, nodes)
    pointer = pointer.to(device)
    pointer_masks = pointer_masks.to(device)

    copy_loss = computer_copy_loss(pro_switch, pointer[:, 1:].to(torch.float).contiguous(), pointer_masks[:, 1:])

    acc_gen = compute_gen_acc(total_pro, gen_outputs[:, 1:].contiguous(), gen_masks[:, :-1])

    # copy_prob = copyer(decoder_input_embeddings, decoder_output_hiddens, pointer[:, 1:])
    # copy_loss = copy_prob.masked_select(pointer_masks[:, 1:]).mean()
    # copy_loss = pro_switch.masked_select(pointer_masks[:, 1:]).mean()
    # encoder_last_hiddenstates = output_dict["encoder_last_hidden_state"][:, -1, :]
    encoder_last_hiddenstates = torch.mean(output_dict["encoder_last_hidden_state"], dim=1)
    sp_rec = sprec(encoder_last_hiddenstates, decoder_output_hiddens, nodes, gen_masks, vocabnode_embedding)
    sprec_loss = computer_sprec_loss(sp_rec, nodes, node_masks)

    recon_positions = recon_positions.to(device)
    rec_logits = reconstructor(recon_positions, output_dict["encoder_hidden_states"][-1])
    rec_loss = compute_ce_loss(rec_logits, recon_relations, recon_masks)

    loss = gen_loss + plm_loss + coverage_loss + rec_loss * config["rec_weight"] \
           + copy_loss * config["cp_weight"] + sprec_loss * config["sprec_weight"]

    plm_optimizer.zero_grad()
    external_optimizer.zero_grad()
    loss.backward()
    external_optimizer.step()
    plm_optimizer.step()

    return plm_loss.item(), gen_loss.item(), rec_loss.item(), copy_loss.item(), \
           coverage_loss.item(), sprec_loss.item(), acc_gen.item()


def run_eval_batch(config, batch, student, plm, sprec, reconstructor, copyer, memory, genproject, device,
                   nodeidx2bartidx):
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
    recon_relations, relations_masks, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks, \
    heads, tails, tri_mask = batch

    kd_description = kd_description.to(device)
    kd_description_masks = kd_description_masks.to(device)

    nodes = nodes.to(device)
    student_embeddings, vocabnode_embedding = student(nodes, edges, types)

    node_masks = node_masks.to(device)
    kd_masks = torch.ne(kd_positions, 0).to(device)

    gen_outputs = gen_outputs.to(device)
    gen_masks = gen_masks.to(device)
    output_dict = plm(input_ids=None,
                      inputs_embeds=student_embeddings,
                      attention_mask=node_masks,
                      decoder_input_ids=gen_outputs[:, :-1],
                      decoder_attention_mask=gen_masks[:, :-1],
                      output_hidden_states=True,
                      labels=gen_outputs[:, 1:].contiguous(),
                      return_dict=True)
    plm_loss = output_dict["loss"]

    hidden_state = output_dict["decoder_hidden_states"]

    tri_mask = tri_mask.to(device)
    recon_relations = recon_relations.to(device)
    recon_masks = recon_masks.to(device)
    qt = memory(hidden_state, heads, tails, tri_mask, recon_relations, student_embeddings)

    decoder_input_embeddings = plm.get_input_embeddings()(gen_outputs[:, :-1])
    decoder_output_hiddens = output_dict["decoder_hidden_states"][-1]

    total_pro, coverage, pro_switch = genproject(decoder_output_hiddens, qt, decoder_input_embeddings, edges,
                                                 nodeidx2bartidx)
    gen_loss = compute_gen_loss(total_pro, gen_outputs[:, 1:].contiguous(), gen_masks[:, :-1])
    coverage_loss = computer_coverage_loss(coverage, nodes)
    pointer = pointer.to(device)
    pointer_masks = pointer_masks.to(device)

    copy_loss = computer_copy_loss(pro_switch, pointer[:, 1:].to(torch.float).contiguous(), pointer_masks[:, 1:])

    acc_gen = compute_gen_acc(total_pro, gen_outputs[:, 1:].contiguous(), gen_masks[:, :-1])

    encoder_last_hiddenstates = output_dict["encoder_last_hidden_state"][:, -1, :]
    sp_rec = sprec(encoder_last_hiddenstates, decoder_output_hiddens, nodes, gen_masks, vocabnode_embedding)
    sprec_loss = computer_sprec_loss(sp_rec, nodes, node_masks)

    # copy_prob = copyer(decoder_input_embeddings, decoder_output_hiddens, pointer[:, 1:])
    # copy_loss = copy_prob.masked_select(pointer_masks[:, 1:]).mean()

    recon_positions = recon_positions.to(device)
    rec_logits = reconstructor(recon_positions, output_dict["encoder_hidden_states"][-1])
    rec_loss = compute_ce_loss(rec_logits, recon_relations, recon_masks)

    return plm_loss.item(), gen_loss.item(), rec_loss.item(), copy_loss.item(), \
           coverage_loss.item(), sprec_loss.item(), acc_gen.item()


def train(config):
    writer_gen_loss = SummaryWriter(comment="generation_loss")
    writer_plm_loss = SummaryWriter(comment="plm_loss")
    writer_graphrec_loss = SummaryWriter(comment="graphrec_loss")
    writer_supervisedcopy_loss = SummaryWriter(comment="supervisedcopy_loss")
    writer_coverage_loss = SummaryWriter(comment="coverage_loss")
    writer_sprec_loss = SummaryWriter(comment="sprec_loss")
    writer_accgen = SummaryWriter(comment="Acc_gen")

    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"])
    vocabs["relation"] = Vocab(config["relation_vocab"])
    with open('./preprocess/nodeidx2bartidx.pkl', 'rb') as file:
        nodeidx2bartidx = pickle.load(file)

    logger.info("Build GCN Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"],
                           config["hidden_size"])
    if config["checkpoint"] == True:
        student.load_state_dict(torch.load(config["external_model"])["student"])
    student.to(device)

    logger.info("Build PLM Model.")
    if config["checkpoint"] == True:
        bart_tokenizer = BartTokenizer.from_pretrained(config["fine_tuned_plm_dir"])
        plm = BartForConditionalGeneration.from_pretrained(config["fine_tuned_plm_dir"])
    else:
        bart_tokenizer = BartTokenizer.from_pretrained(config["plm_dir"])
        plm = BartForConditionalGeneration.from_pretrained(config["plm_dir"])
    plm.to(device)

    logger.info("Build Reconstructor Model.")
    reconstructor = GraphReconstructor(vocabs["relation"].size(), config["hidden_size"])
    if config["checkpoint"] == True:
        reconstructor.load_state_dict(torch.load(config["external_model"])["reconstructor"])
    reconstructor.to(device)

    logger.info("Build Copy Model.")
    copyer = GraphPointer(config["embedding_size"], config["hidden_size"])
    if config["checkpoint"] == True:
        copyer.load_state_dict(torch.load(config["external_model"])["copyer"])
    copyer.to(device)

    logger.info("Build Memory Module.")
    memory = Memory_Shift(config["hidden_size"], config["embedding_size"], vocabs["relation"].size(),
                          config["relation_embedding"])
    if config["checkpoint"] == True:
        memory.load_state_dict(torch.load(config["external_model"])["memory"])
    memory.to(device)

    logger.info("Build Generation Projection Model.")
    genproject = GenProjection(config["hidden_size"], vocabs["node"].size())
    if config["checkpoint"] == True:
        genproject.load_state_dict(torch.load(config["external_model"])["genproject"])
    genproject.to(device)

    logger.info("Build Static Plan Reconstruction Model.")
    sprec = StaticPlanReconstruction(vocabs["node"].size(), config["hidden_size"], config["embedding_size"])
    if config["checkpoint"] == True:
        sprec.load_state_dict(torch.load(config["external_model"])["sprec"])
    sprec.to(device)

    plm_parameters = [p for p in plm.parameters() if p.requires_grad]
    for p in memory.parameters():
        if p.requires_grad:
            plm_parameters.append(p)
    for p in genproject.parameters():
        if p.requires_grad:
            plm_parameters.append(p)
    plm_optimizer = build_optimizer(plm_parameters, config["plm_learner"], config["plm_lr"], config)

    external_parameters = []
    for p in student.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    for p in reconstructor.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    for p in copyer.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    for p in sprec.parameters():
        if p.requires_grad:
            external_parameters.append(p)
    external_optimizer = build_optimizer(external_parameters, config["external_learner"], config["external_lr"], config)

    logger.info("Create training dataset.")
    train_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples=config["num_samples"], usage="train"),
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    logger.info("Create validation dataset.")
    valid_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples="all", usage="valid"),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    gen_losses = []
    copy_losses = []
    rec_losses = []
    best_gen_loss = None
    iteration = 0
    for epoch_idx in range(config["start_epoch"], config["epochs"]):

        student.train()
        plm.train()
        reconstructor.train()
        copyer.train()
        memory.train()
        genproject.train()
        train_gen_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            plm_loss, gen_loss, rec_loss, copy_loss, coverage_loss, sprec_loss, acc_gen = run_train_batch(config, batch,
                                                                                                          student,
                                                                                                          plm, sprec,
                                                                                                          reconstructor,
                                                                                                          copyer,
                                                                                                          memory,
                                                                                                          genproject,
                                                                                                          plm_optimizer,
                                                                                                          external_optimizer,
                                                                                                          device,
                                                                                                          nodeidx2bartidx)

            logger.info(
                "Epoch {} batch {}:  Gen loss {} PLM loss {} Rec loss {} "
                "Copy loss {} Coverage_loss {} SP rec loss {} Acc_gen {}.".format(epoch_idx, batch_idx,
                                                                                  gen_loss, plm_loss,
                                                                                  rec_loss, copy_loss,
                                                                                  coverage_loss, sprec_loss,
                                                                                  acc_gen))
            writer_gen_loss.add_scalar("gen_loss", gen_loss, global_step=iteration)
            writer_plm_loss.add_scalar("plm_loss", plm_loss, global_step=iteration)
            writer_coverage_loss.add_scalar("coverage_loss", coverage_loss, global_step=iteration)
            writer_supervisedcopy_loss.add_scalar("supervisedcopy_loss", copy_loss, global_step=iteration)
            writer_sprec_loss.add_scalar("sprec_loss", sprec_loss, global_step=iteration)
            writer_graphrec_loss.add_scalar("graphrec_loss", rec_loss, global_step=iteration)
            writer_accgen.add_scalar("accgen", acc_gen, global_step=iteration)
            iteration = iteration + 1

            train_gen_loss += gen_loss

            gen_losses.append(gen_loss)
            rec_losses.append(rec_loss)
            copy_losses.append(copy_loss)

        train_gen_loss /= len(train_dataloader)
        train_ppl = np.exp(train_gen_loss)
        training_time = format_time(time.time() - t0)
        logger.info("\n\nEpoch {}: training generation loss {} perplexity {} time {}.\n".format(epoch_idx,
                                                                                                train_gen_loss,
                                                                                                train_ppl,
                                                                                                training_time))

        with torch.no_grad():
            # teacher.eval()
            student.eval()
            memory.eval()
            genproject.eval()
            plm.eval()
            reconstructor.eval()
            copyer.eval()
            valid_gen_loss = 0
            t0 = time.time()
            for batch in valid_dataloader:
                plm_loss, gen_loss, rec_loss, copy_loss, \
                coverage_loss, sprec_loss, acc_gen = run_eval_batch(config, batch, student, plm, sprec, reconstructor,
                                                                    copyer, memory, genproject, device,
                                                                    nodeidx2bartidx)
                valid_gen_loss += gen_loss

            valid_gen_loss /= len(valid_dataloader)
            valid_ppl = np.exp(valid_gen_loss)
            valid_time = format_time(time.time() - t0)
            logger.info("\n\nEpoch {}: validation generation loss {} perplexity {} time {}.\n".format(epoch_idx,
                                                                                                      valid_gen_loss,
                                                                                                      valid_ppl,
                                                                                                      valid_time))

        if best_gen_loss is None or valid_gen_loss <= best_gen_loss:
            output_dir = '{}-{}-{}-update'.format(config["dataset"], config["num_samples"], str(epoch_idx))
            saved_path = os.path.join("./ckpt", output_dir)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            # save pretrained language model
            model_to_save = plm.module if hasattr(plm, 'module') else plm
            model_to_save.save_pretrained(saved_path)
            bart_tokenizer.save_pretrained(saved_path)

            # save knowledge adapter
            torch.save(config, os.path.join(saved_path, 'training_configurations.bin'))
            torch.save({"student": student.state_dict(),
                        "reconstructor": reconstructor.state_dict(),
                        "copyer": copyer.state_dict(),
                        "genproject": genproject.state_dict(),
                        "memory": memory.state_dict(),
                        "sprec": sprec.state_dict()},
                       os.path.join(saved_path, 'external.bin'))
            logger.info("Save student, reconstructor, copyer and fine-tuned PLM model into {}.".format(saved_path))

            best_gen_loss = valid_gen_loss


def test(config):
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    init_seed(config["seed"], config["reproducibility"])
    device = init_device(config)

    logger.info("Build node and relation vocabularies.")
    vocabs = dict()
    vocabs["node"] = Vocab(config["node_vocab"])
    vocabs["relation"] = Vocab(config["relation_vocab"])
    with open('./preprocess/nodeidx2bartidx.pkl', 'rb') as file:
        nodeidx2bartidx = pickle.load(file)

    logger.info("Build GCN Model.")
    student = GraphEncoder(vocabs["node"].size(), vocabs["relation"].size(),
                           config["gnn_layers"], config["embedding_size"], config["node_embedding"],
                           config["hidden_size"])
    student.load_state_dict(torch.load(config["external_model"])["student"])
    student.to(device)

    logger.info("Build PLM Model.")
    bart_tokenizer = BartTokenizer.from_pretrained(config["fine_tuned_plm_dir"])
    plm = BartForConditionalGeneration.from_pretrained(config["fine_tuned_plm_dir"])
    plm.to(device)

    logger.info("Build Memory Module.")
    memory = Memory_Shift(config["hidden_size"], config["embedding_size"], vocabs["relation"].size(),
                          config["relation_embedding"])
    memory.load_state_dict(torch.load(config["external_model"])["memory"])
    memory.to(device)

    logger.info("Build Generation Projection Model.")
    genproject = GenProjection(config["hidden_size"], vocabs["node"].size())
    genproject.load_state_dict(torch.load(config["external_model"])["genproject"])
    genproject.to(device)

    logger.info("Create testing dataset.")
    test_dataloader = DataLoader(
        S2SDataset(data_dir=config["data_dir"], dataset=config["dataset"],
                   tokenizer=bart_tokenizer, node_vocab=vocabs["node"], relation_vocab=vocabs["relation"],
                   num_samples="all", usage="test"),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn_graph_text,
        pin_memory=True)

    student.eval()
    memory.eval()
    genproject.eval()
    plm.eval()
    idx = 0
    generated_text = []
    reference_text = []
    with torch.no_grad():
        for batch in test_dataloader:
            nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
            recon_relations, relations_masks, recon_positions, recon_masks, gen_outputs, gen_masks, \
            pointer, pointer_masks, heads, tails, tri_mask = batch

            # kd_description = kd_description.to(device)
            # kd_description_masks = kd_description_masks.to(device)
            # output_dict = teacher(kd_description,
            #                       attention_mask=kd_description_masks,
            #                       output_hidden_states=True,
            #                       return_dict=True)
            # positions = kd_positions.unsqueeze(-1).expand(-1, -1, output_dict["encoder_last_hidden_state"].size(-1)).to(device)
            # teacher_embeddings = torch.gather(output_dict["encoder_last_hidden_state"], dim=1, index=positions).detach()

            nodes = nodes.to(device)
            student_embeddings, vocabnode_embedding = student(nodes, edges, types)

            node_masks = node_masks.to(device)
            gen_outputs = gen_outputs.to(device)
            gen_masks = gen_masks.to(device)
            # output_dict = plm.generate(input_ids=None,
            #                            inputs_embeds=student_embeddings,
            #                            attention_mask=node_masks,
            #                            num_beams=1,
            #                            max_length=config["max_seq_length"],
            #                            output_hidden_states=True,
            #                            output_scores=True,
            #                            return_dict_in_generate=True,
            #                            early_stopping=True)
            # hidden_state = tuple([i.squeeze() for i in
            #                       torch.stack([torch.stack([each_layer for each_layer in each_generated_token], dim=0)
            #                                    for each_generated_token in output_dict["decoder_hidden_states"]],
            #                                   dim=0).squeeze().permute(1, 2, 0, 3).chunk(7, dim=0)])
            output_dict = plm(input_ids=None,
                              inputs_embeds=student_embeddings,
                              attention_mask=node_masks,
                              decoder_input_ids=gen_outputs[:, :-1],
                              decoder_attention_mask=gen_masks[:, :-1],
                              output_hidden_states=True,
                              labels=gen_outputs[:, 1:].contiguous(),
                              return_dict=True)
            hidden_state = output_dict["decoder_hidden_states"]

            tri_mask = tri_mask.to(device)
            recon_relations = recon_relations.to(device)
            qt = memory(hidden_state, heads, tails, tri_mask, recon_relations, student_embeddings)

            decoder_input_embeddings = plm.get_input_embeddings()(gen_outputs[:, :-1])
            decoder_output_hiddens = hidden_state[-1]
            total_pro, coverage, pro_switch = genproject(decoder_output_hiddens, qt, decoder_input_embeddings, edges,
                                                         nodeidx2bartidx)
            generated_ids = torch.max(total_pro, dim=-1)[1]

            generated = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            reference = bart_tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            generated_text.extend(generated)
            reference_text.extend(reference)

            idx += 1
            logger.info("Finish {}-th example.".format(idx))

    assert len(generated_text) == len(reference_text)
    saved_file = "{}-{}.res".format(config["dataset"], config["num_samples"])
    saved_file_path = os.path.join(config["output_dir"], saved_file)
    fout = open(saved_file_path, "w")
    for i in range(len(generated_text)):
        fout.write("Generated text: " + generated_text[i].strip() + "\n")
        fout.write("Reference text: " + reference_text[i].strip() + "\n")
    fout.close()


def main():
    config = read_configuration("config.yaml")

    if config["mode"] == "train":
        train(config)
    else:
        test(config)


if __name__ == '__main__':
    main()
