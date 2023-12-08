from load_data import load_train_data, Collater_base, Collater_frsq, Collater_sqsq
from joint import train_desq
from utils import*

def save_model(model, epoch):
    model.save_pretrained(args.save_model_path + "/{}/epoch{}/".format(args.m_code, epoch))
    logger.info("Saved the {} model of epoch:{}".format(args.m_code, epoch))


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    batch_step = len(train_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        try:
            inputs = globals().get("batch_buil_" + args.m_code)(batch_data)
            loss = model(**inputs).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)
            del loss, batch_data, inputs

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


def valid_epoch(model, valid_dataloader, epoch):
    model.eval()
    total_loss = 0
    batch_step = len(valid_dataloader)

    for batch_idx, batch_data in enumerate(tqdm(valid_dataloader)):
        try:
            inputs = globals().get("batch_buil_" + args.m_code)(batch_data)
            loss = model(**inputs).loss

            total_loss += float(loss)
            del loss, batch_data, inputs

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


def train(model, processor, train_data, valid_data):
    patience = 0
    best_val_loss = float('Inf')

    if args.augment and args.m_code == "frsq":
        collate_fn = globals().get("Collater_" + args.m_code)(processor, args.device, **make_kwargs(args))
    else:
        collate_fn = globals().get("Collater_" + args.m_code)(processor, args.device)

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
    optimizer = transformers.AdamW(list(model.parameters()), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warm_step, t_total)

    for epoch in range(1, args.epochs+1):
        train_epoch(model, train_dataloader, optimizer, scheduler, epoch)

        if valid_data is not None:
            val_loss = valid_epoch(model, valid_dataloader, epoch)

            if val_loss < best_val_loss:
                # Save model and variational model
                save_model(model, epoch)
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
                save_model(model, epoch)

    logger.info('Training finished')


def main():
    device_assign(args, logger)
    train_data, valid_data = load_train_data(args.m_code, args.train_csv_path, args.num_valid, args.d_rms, logger)

    if args.m_code in ["base", "frsq", "sqsq"]:
        if args.m_code is "sqsq":
            processor = AutoTokenizer.from_pretrained(args.gptf_pretrained, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(args.gptf_pretrained)
            logger.info("Initialized LLM from the pretrained weight for Toyama text -> Tokyo text (alone)")
        else:
            processor = Wav2Vec2Processor.from_pretrained(args.xlsr_pretrained)
            model = Wav2Vec2ForCTC.from_pretrained(args.xlsr_pretrained)

            if args.m_code is "base":
                logger.info("Initialized the baseline model from the pretrained weight")
            if args.m_code is "frsq":
                logger.info("Initialized Wav2Vec from the pretrained weight for Toyama audio -> Toyama text")

        model = to_device(args.device, logger, model)
        train(model, processor, train_data, valid_data)

    if args.m_code is "desq":
        processor = Wav2Vec2Processor.from_pretrained(args.xlsr_pretrained)
        wav = Wav2Vec2ForCTC.from_pretrained(args.frsq_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.gptf_pretrained, use_fast=False)
        llm = AutoModelForCausalLM.from_pretrained(args.sqsq_checkpoint)
        logger.info("Restored LLM, Wav2Vec from the checkpoint for Toyama text -> Tokyo text (joint)")

        wav, llm = to_device(args.device, logger, wav, llm)
        train_desq(wav, llm, processor, tokenizer, train_data, valid_data, args, logger)


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()