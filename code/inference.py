from load_data import load_test_data
from utils import*

class Base():
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(args.xlsr_pretrained)
        model = Wav2Vec2ForCTC.from_pretrained(args.base_checkpoint)
        self.model = to_device(args.device, logger, model).eval()
        logger.info("Restored the baseline model from the check point")

    def inference(self, speech):
        input = self.processor(speech, sampling_rate=16_000, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = self.model(**input).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]


class Frsq(Base):
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(args.xlsr_pretrained)
        model = Wav2Vec2ForCTC.from_pretrained(args.frsq_checkpoint)
        self.model = to_device(args.device, logger, model).eval()
        logger.info("Restored our model from the check point for Toyama audio -> Toyama text")


class Sqsq():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(args.gptf_pretrained, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.sqsq_checkpoint)
        self.generator = pipeline("text-generation", model=model.eval(), tokenizer=tokenizer, device=0)
        logger.info("Restored our model from the check point for Toyama text -> Tokyo text")

    def inference(self, text):
        pred = self.generator(text, max_length=args.maxresp_len, do_sample=args.do_sample, temperature=args.temperature,
                              top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repeti_pena,
                              num_beams=args.num_beams, num_return_sequences=args.num_ret_seq)

        return pred[0]["generated_text"].split("\nシステム:")[-1]


class Desq(Sqsq):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(args.gptf_pretrained, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(args.desq_checkpoint)
        self.generator = pipeline("text-generation", model=model.eval(), tokenizer=tokenizer, device=0)
        logger.info("Restored our model from the check point for Toyama text -> Tokyo text")


class Dtd1():
    def __init__(self):
        self.frsq = Frsq()
        self.sqsq = Sqsq()

    def inference(self, speech):
        input_text = self.frsq.inference(speech)
        text = "「{}」を標準語に変換してください。".format(input_text.strip())
        prompt = f"ユーザー: {text}\nシステム:"
        pred = self.sqsq.inference(prompt)

        return pred, input_text


class Dtd2():
    def __init__(self):
        self.frsq = Frsq()
        self.desq = Desq()

    def inference(self, speech):
        input_text = self.frsq.inference(speech)
        text = "「{}」を標準語に変換してください。".format(input_text.strip())
        prompt = f"ユーザー: {text}\nシステム:"
        pred = self.desq.inference(prompt)

        return pred, input_text


def main():
    device_assign(args, logger)
    test_data = load_test_data(args.m_code, args.test_csv_path, args.d_rms, logger)

    if not os.path.exists(args.save_resul_path):
        os.makedirs(args.save_resul_path)

    if args.m_code is "base":
        base = Base()
        file = codecs.open(os.path.join(args.save_resul_path, "base_predicted"), 'w', 'utf8')

        for idx, speech in enumerate(test_data["speech"]):
            file.write("- expect: " + test_data["Tokyo"][idx] + "\n")
            file.write("- predic: " + base.inference(speech) + "\n\n")

    if args.m_code in ["dtd1", "dtd2"]:
        id = args.m_code[-1]
        dtd = globals().get("Dtd" + id)()
        file = codecs.open(os.path.join(args.save_resul_path, "dtd"+id+"_predicted"), 'w', 'utf8')

        for idx, speech in enumerate(test_data["speech"]):
            pred, text = dtd.inference(speech)
            file.write("- Tokyo : " + test_data["Tokyo"][idx] + "\n")
            file.write("- Toyama: " + test_data["Toyama"][idx] + "\n")
            file.write("- au->tx: " + text + "\n")
            file.write("- predic: " + pred + "\n\n")

    if args.m_code is "rule":
        return

    file.close()
    logger.info('Inference finished')


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)
    main()