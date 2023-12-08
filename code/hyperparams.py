import argparse

def set_args():
    parser = argparse.ArgumentParser()

    # ============= common path ============= #
    parser.add_argument('--log_path',        default='../save_load/log/log.log',        type=str)
    parser.add_argument('--test_csv_path',   default='../dataset/test.csv',             type=str)
    parser.add_argument('--train_csv_path',  default='../dataset/train.csv',            type=str)
    parser.add_argument('--save_model_path', default='../save_load/model',              type=str)
    parser.add_argument('--save_resul_path', default='../save_load/result',             type=str)
    parser.add_argument('--xlsr_pretrained', default='../save_load/pre_weight/xlsr',    type=str)
    parser.add_argument('--gptf_pretrained', default='../save_load/pre_weight/gpt-sft', type=str)
    parser.add_argument('--base_checkpoint', default='../save_load/model/base/epoch10', type=str)
    parser.add_argument('--frsq_checkpoint', default='../save_load/model/frsq/epoch55', type=str)
    parser.add_argument('--desq_checkpoint', default='../save_load/model/desq/epoch55', type=str)
    parser.add_argument('--sqsq_checkpoint', default='../save_load/model/sqsq/epoch20', type=str)

    # =============== training ============== #
    parser.add_argument('--device',     default='0',    type=str)
    parser.add_argument('--batch_size', default=20,     type=int)
    parser.add_argument('--epochs',     default=55,     type=int)
    parser.add_argument('--lr',         default=1e-5,   type=float)
    parser.add_argument('--warm_step',  default=1000,   type=int)
    parser.add_argument('--n_worker',   default=1,      type=int)

    # ============== scheduler ============== #
    parser.add_argument('--hybrid',     default=False,  type=bool,  help='Whether use label for desq')
    parser.add_argument('--time_scale', default=1.5,    type=float, help='Time scale of audio for desq')
    parser.add_argument('--augment',    default=True,  type=bool,  help='If use a data augment for frsq')
    parser.add_argument('--num_frame',  default=100,    type=int,   help='Number of mask frame for audio')
    parser.add_argument('--num_frequ',  default=20,     type=int,   help='Number of mask frequen for audio')
    parser.add_argument('--n_mels',     default=128,    type=int,   help='Number of filter bands to generate')
    parser.add_argument('--n_fft',      default=512,    type=int,   help='Number of FFT components result mel')
    parser.add_argument('--hop_length', default=128,    type=int,   help='Number of samples between the frames')
    parser.add_argument('--n_iter',     default=50,     type=int,   help='Number of iteration for melspect to audio')
    parser.add_argument('--d_rms',      default=0.025,  type=float, help='RMS for audio regularization, set 0 if not use')
    parser.add_argument('--num_valid',  default=0,      type=int,   help='Number of samples use for valid, set 0 if not valid')
    parser.add_argument('--patience',   default=3,      type=int,   help='Number of patience times, set num_valid >= 1 if use')
    parser.add_argument('--niter_save', default=50,     type=int,   help='Save model after set epoch, set num_valid as 0 if use')
    parser.add_argument('--a_code',     default='both', type=str,   help='Must set as the one of in list [both, eith, time, freq]\
                                                                          time : Use the time masking only          \
                                                                          freq : Use the frequency masking only     \
                                                                          both : Use both time and frequency masking\
                                                                          eith : Use one of time or frequency masking')
    parser.add_argument('--m_code',     default='frsq', type=str,   help='Must set as the one of in [base, frsq, sqsq, desq, dtd1, dtd2, rule]\
                                                                          ============= for training or inference ==========\
                                                                          base : Toyama audio -> Tokyo text                 \
                                                                          =================== for training =================\
                                                                          frsq : Toyama audio -> Toyama text                \
                                                                          sqsq : Toyama text  -> Tokyo text (train individu)\
                                                                          desq : Toyama text  -> Tokyo text (joint training)\
                                                                          =================== for inference ================\
                                                                          dtd1 : Toyama audio -> Tokyo text (use above sqsq)\
                                                                          dtd2 : Toyama audio -> Tokyo text (use above desq)\
                                                                          rule : Toyama audio -> Tokyo text')

    # =============== inference =============== #
    parser.add_argument('--maxresp_len', default=100,  type=int)
    parser.add_argument('--repeti_pena', default=1.1,  type=float)
    parser.add_argument('--temperature', default=1.0,  type=float)
    parser.add_argument('--top_p',       default=0.9,  type=float)
    parser.add_argument('--top_k',       default=0,    type=int)
    parser.add_argument('--num_beams',   default=1,    type=int)
    parser.add_argument('--num_ret_seq', default=1,    type=int)
    parser.add_argument('--do_sample',   default=True, type=int)

    return parser.parse_args()