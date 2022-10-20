# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import torch
import transformers
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from options import FiDOptions

import src.slurm
import src.util
import src.evaluation
from data import FiDDataset, Collator
from models import FiDT5

from torch.utils.tensorboard import SummaryWriter




def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=2,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    last_step = step
    opt.total_steps = opt.total_steps if opt.total_steps is not None else opt.epochs * len(train_dataloader)
    status_bar = tqdm(total=opt.total_steps)
    early_stop_count = 0
    while step < opt.total_steps and early_stop_count < opt.early_stop_count:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            if step % 20 == 0:
                status_bar.update(step-last_step)
                last_step = step
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em < best_dev_em:
                        early_stop_count = 0
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    else:
                        early_stop_count += 1
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            # if opt.is_main and step % opt.save_freq == 0:
            #     src.util.save(model, optimizer, scheduler, step, best_dev_em,
            #               opt, checkpoint_path, f"step-{step}")
            if step >= opt.total_steps or early_stop_count >= opt.early_stop_count:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=2,
        collate_fn=collator
    )
    model.eval()
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        curr_loss = 0
        for batch in dataloader:
            (idx, labels, _, context_ids, context_mask) = batch
    
            loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]
            loss = src.util.average_main(loss, opt)
            curr_loss += loss.item()
            
    return curr_loss/len(dataloader)

if __name__ == "__main__":
    options = FiDOptions()
    options.add_model_specific_options()
    options.add_optim_options()
    options.add_train_options()
    opt = options.parse()
    if opt.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_dataset = FiDDataset(
        opt.train_data, 
        opt.n_context,
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
        no_sent=opt.no_sent,
        no_path=opt.no_path,
        duplicate_sample=opt.duplicate_sample
    )
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_dataset = FiDDataset(
        opt.eval_data, 
        opt.n_context,
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
        no_sent=opt.no_sent,
        no_path=opt.no_path,
        duplicate_sample=opt.duplicate_sample
    )

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        # model = model.to(opt.local_rank)
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 100.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
