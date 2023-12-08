from load_data import Collater_desq, format_desp
from utils import*

def save_model(model, epoch, args, logger):
    model.save_pretrained(args.save_model_path + "/{}/epoch{}/".format(args.m_code, epoch))
    logger.info("Saved the {} model of epoch:{}".format(args.m_code, epoch))


def train_epoch(wav, llm, processor, tokenizer, train_dataloader, optimizer, scheduler, epoch, args, logger):
    wav.eval()
    llm.train()
    total_loss = 0
    batch_step = len(train_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        try:
            Tokyo, Toyama, inputs = batch_buil_desq(batch_data, "wav")

            if np.random.rand() < 0.2 and args.hybrid:
                pair = format_desp(Tokyo, Toyama, tokenizer, args.device)
            else:
                with torch.no_grad():
                    logits = wav(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                pair = format_desp(Tokyo, transcription, tokenizer, args.device)

            inputs = batch_buil_desq(pair, "llm")
            loss = llm(**inputs).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)
            del loss, batch_data, inputs, pair

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    nll = total_loss / batch_step
    logger.info("Training epoch:{} Loss:{}".format(epoch, nll))


def valid_epoch(wav, llm, processor, tokenizer, valid_dataloader, epoch, args, logger):
    wav.eval()
    llm.eval()
    total_loss = 0
    batch_step = len(valid_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(valid_dataloader)):
        try:
            Tokyo, _, inputs = batch_buil_desq(batch_data, "wav")

            with torch.no_grad():
                logits = wav(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            pair = format_desp(Tokyo, transcription, tokenizer, args.device)
            inputs = batch_buil_desq(pair, "llm")
            loss = llm(**inputs).loss

            total_loss += float(loss)
            del loss, batch_data, inputs, predicted_ids, transcription, pair

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    nll = total_loss / batch_step
    logger.info("Validating epoch:{} Loss:{}".format(epoch, nll))

    return nll


def train_desq(wav, llm, processor, tokenizer, train_data, valid_data, args, logger):
    patience = 0
    best_val_loss = float('Inf')

    wav = param_freeze(wav)
    collate_fn = Collater_desq(processor, args.device, **make_kwargs(args))

    train_dataloader = \
    DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn)

    if valid_data is not None:
        valid_dataloader = \
        DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.n_worker, collate_fn=collate_fn)
        logger.info('Starting training with validation')
    else:
        logger.info('Starting training without validation')

    # ========== train ========== #
    t_total = len(train_dataloader) * args.epochs
    optimizer = transformers.AdamW(list(llm.parameters()), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)

    for epoch in range(1, args.epochs+1):
        train_epoch(wav, llm, processor, tokenizer, train_dataloader, optimizer, scheduler, epoch, args, logger)

        if valid_data is not None:
            val_loss = valid_epoch(wav, llm, processor, tokenizer, valid_dataloader, epoch, args, logger)

            if val_loss < best_val_loss:
                # Save model and variational model
                save_model(llm, epoch, args, logger)
                best_val_loss = val_loss
                patience = 0
            else:
                # This variable is useless when early_stop is False
                patience = patience + 1

            if args.patience < patience:
                logger.info("Early stop due to run out of patience")
                break
        else:
            if args.niter_save <= epoch:
                save_model(llm, epoch, args, logger)

    logger.info('Training finished')