import os
import argparse
import sys
import subprocess
import pickle

class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)


def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


def postprocess_opts(parser):
    group = parser.add_argument_group('Infer')
    group.add_argument("-infer_script",
                       type=str,
                       default="../translate.py",
                       help="The address of the translate.py")
    group.add_argument("-src",
                       type=str,
                       help="The address of the source file that will be translated by model."
                            "Each line contains a single example.")
    group.add_argument("-tgt",
                       type=str,
                       help="The address of the gold translated sentence files")
    group.add_argument("-infer_param",
                       type=str,
                       default="",
                       help="Additional param while running."
                            "Run will be like `python translate.py {infer_param}`. "
                            "Example: `-verbose -beam_size 5` "
                            "see the link for the params, "
                            "http://opennmt.net/OpenNMT-py/options/translate.html")

    group = parser.add_argument_group('Dataset')
    group.add_argument("-dir",
                       type=str,
                       required=True,
                       help="folder address where the data files exists. "
                            "data file pattern: $NAME_acc_XX.YY_ppl_XX.YY_eZZ.pt "
                            "ex: luong.dot_acc_66.17_ppl_5.77_e11.pt")
    group.add_argument("-name",
                       type=str,
                       required=True,
                       help="Name of the experiment. "
                            "data file pattern: $NAME_acc_XX.YY_ppl_XX.YY_eZZ.pt "
                            "ex: luong.dot_acc_66.17_ppl_5.77_e11.pt, here $NAME='luong.dot'.")

    group = parser.add_argument_group('Run-type')
    group.add_argument("-select_max_acc",
                        action='store_true',
                        help="Select one the model based on maximum acc, lowest ppl within the -dir/models folder.")
    group.add_argument("-select_min_ppl",
                       action='store_true',
                       help="Select one the model based on lowest ppl, maximum acc within the -dir/models folder.")
    group.add_argument("-select_after_epoch",
                       type=int,
                       default=-1,
                       help="Only translate if checkpoint is greater than the -select_after_epoch.")
    group.add_argument("-verbose",
                       action='store_true',
                       help="Print the additional information")
    # mutual_param = group.add_mutually_exclusive_group(required=True)
    # mutual_param.add_argument("-wait_for_checkpoint",
    #                            action='store_true',
    #                            help="Wait for checkpoint to the generate.")
    # mutual_param.add_argument("-process_id",
    #                            type=int,
    #                            default=0,
    #                            help="Process id that the file will monitor.")

    group = parser.add_argument_group('Tokenizer Post-processing')
    group.add_argument("-moses_tok_script",
                       type=str,
                       default="tokenizer.perl",
                       help="The address of the script of moses tokenizer")
    group.add_argument("-moses_detok_script",
                       type=str,
                       default="detokenizer.perl",
                       help="The address of the script of moses detokenizer")
    group.add_argument("-src_lang",
                       type=str,
                       default="en",
                       help="short form of source language")
    group.add_argument("-tgt_lang",
                       type=str,
                       default="ms",
                       help="short form of target language.")
    group.add_argument("-blue_score_script",
                        type=str,
                        help="The address of the script that will calculate the blue score."
                             "tokenizer option: multi-bleu-detok.perl, multi-bleu.perl")
    group.add_argument("-bpe_apply_script",
                       type=str,
                       default="apply_bpe.py",
                       help="The address of the script that will calculate "
                            "the bype-pair encodding oif a sentence.")
    group.add_argument("-bpe_vocab_address",
                        type=str,
                        default="",
                        help="If BPE process is applied in dataset preprocessing"
                             " and postprocessing only if a valid bpe vocab address is given")

    group = parser.add_argument_group('Chart')
    group.add_argument("-chart_folder",
                        type=str,
                        default="chart",
                        help="short identifier of the target language. ex: ms")
    group.add_argument("-chart_name",
                        type=str,
                        default="chart",
                        help="short identifier of the target language. ex: ms")


def assert_address(opt):
    """
    Checks is a address exists or not.
    :param opt: parser
    :return: return 0 on success else raise exception.
    """
    try:
        assert os.path.exists(opt.dir)
        assert os.path.exists(opt.blue_score_script)
        assert os.path.exists(opt.moses_tok_script)
        assert os.path.exists(opt.infer_script)
        assert os.path.exists(opt.src)
    except AssertionError as e:
        print("opt.dir :", opt.dir)
        print("opt.blue_score_script :", opt.blue_score_script)
        print("opt.infer_script :", opt.infer_script)
        print("opt.src :", opt.src)
        print(e)
        raise
    return 0


def assert_source(opt):
    """
    read the source file and ensures there's not empty line.
    :param opt: parser
    :return: return zero on success else raise exception.
    """
    file_ptr = ""
    try:
        file_ptr = open(opt.src, "r")
    except Exception as e:
        print("Can not read {0} file.\n{1}".format(opt.src, e))
    cnt = 0
    for line in file_ptr:
        if line == "":
            raise Exception("Souce file contains empty line.")
        cnt += 1
    if opt.verbose:
        print("Total number of example in the source file: {0}".format(cnt))
    return 0


def calc_us_in_name(name):
    """
    calculate how many underscore is in name.
    :param name: a string.
    :return: a number
    """
    cnt = 0
    for ch in name:
        if ch == '_':
            cnt += 1
    return cnt


class Checkpoints:
    """
    class for saving information of a model.
    attributes:
    address: full address of the checkpoint.
    name: name of the model
    acc: accuracy of the model. (more is better)
    ppl: perplexity of a model. (less is better)
    epoch: epoch at which the model saved
    """
    def __init__(self, address, name, acc, ppl, epoch):
        self.address = address
        self.name = name
        self.acc = acc
        self.ppl = ppl
        self.epoch = epoch
        self.bleu = ''

    @staticmethod
    def str2epoch(_epoch_token):
        try:
            epoch = float(_epoch_token[1:-3])
        except Exception as e:
            print(e, "\n Canot retrieve epoch number from {0} sting."
                  .format(_epoch_token))
            raise
        return epoch


def valid_model_name(name, opt, step=0):
    """
    chcek if a sting is valid checkpoint name.
    naming pattern: $NAME_acc_XX.YY_ppl_XX.YY_eZZ.pt
    :param name: name of the checkpoint
    :param opt: parser
    :return: return True if the name if valid else False.
    """
    parts = name.strip().split('_')
    if len(parts) != (6+step):
        return False
    tmp = parts[0:0+step+1]
    _name = ''
    for i in tmp:
        _name += i + '_'
    _name = _name[0:-1]
    if _name != opt.name:
        return False
    if parts[1+step] != 'acc':
        return False
    try:
        if type(float(parts[2+step])) is not float:
            return False
    except ValueError as e:
        if opt.verbose:
            print("{0} is not a valid model address.\nException details: {1}".
                  format(name, e))
        return False
    if parts[3+step] != 'ppl':
        return False
    try:
        if type(float(parts[4+step])) is not float:
            return False
    except Exception as e:
        if opt.verbose:
            print("{0} is not a valid model address.\nException details: {1}".
                  format(name, e))
        return False
    if parts[5+step][-3:] != '.pt':
        return False
    if parts[5+step][0:1] != 'e':
        return False
    try:
        if type(int(parts[5+step][1:-3])) is not int:
            return False
    except Exception as e:
        if opt.verbose:
            print("{0} is not a valid model address.\nException details: {1}".
                  format(name, e))
        return False
    return True


def retrieve_model_list(opt):
    """
    retrive the model information from form a directory.
    :param opt: parser
    :return: list of Checkpoint object.
    """
    files = os.listdir(os.path.join(opt.dir, 'models'))
    files.sort()
    # file name format "address/$NAME_acc_XX.YY_ppl_XX.YY_eZZ.pt"
    valid_address = []
    for address in files:
        name = os.path.basename(address)
        step = calc_us_in_name(name) - 5
        if valid_model_name(name, opt, step=step):
            lst = name.strip().split('_')
            valid_address.append(
                Checkpoints(
                    os.path.join(
                        os.path.join(opt.dir, 'models'),
                        address
                    ),
                    str(lst[0+step]),
                    float(lst[2+step]),
                    float(lst[4+step]),
                    Checkpoints.str2epoch(lst[5+step])
                )
            )
    try:
        assert len(valid_address) != 0
    except AssertionError as e:
        print("{0}\nNo valid model found in {1} with name={2}."
              .format(e, opt.dir, opt.name))
        raise
    return valid_address


def select_checkpoint(checkpoint_list, opt):
    ret = []
    if opt.select_max_acc:
        mx_idx = mx_acc = 0
        mn_ppl = 1e6
        for (idx, chk_pt) in enumerate(checkpoint_list):
            if chk_pt.acc > mx_acc:
                mx_idx = idx
                mx_acc = chk_pt.acc
                mn_ppl = chk_pt.ppl
            if chk_pt.acc == mx_acc and \
                    chk_pt.ppl < mn_ppl:
                mx_idx = idx
                mn_ppl = chk_pt.ppl
        ret += checkpoint_list[mx_idx:mx_idx+1]
    if opt.select_min_ppl:
        mn_idx = mx_acc = 0
        mn_ppl = 1e6
        for (idx, chk_pt) in enumerate(checkpoint_list):
            if chk_pt.ppl < mn_ppl:
                mn_idx = idx
                mx_acc = chk_pt.acc
                mn_ppl = chk_pt.ppl
            if chk_pt.ppl == mn_ppl and \
                    chk_pt.acc > mx_acc:
                mn_idx = idx
                mx_acc = chk_pt.acc

        ret += checkpoint_list[mn_idx:mn_idx + 1]

    if opt.select_min_ppl or opt.select_max_acc:
        return ret

    ret = sorted(checkpoint_list, key=lambda x: x.epoch)
    return ret


def create_address(address, opt):
    """
    given a checkpoint address, it creates a new folder named pred
    (if is not there) into the checkpoint folder and creates the
    corresponding output file address adding `.pred`  at the end
    of the checkpoint name.
    :param address: address of a checkpoint
    :return: returns the prediction file address that will be
             created
    """
    _dir = os.path.dirname(address)
    file_name = os.path.basename(address)
    folder_name = os.path.join(_dir, "pred")
    os.makedirs(folder_name, exist_ok=True)
    if opt.verbose:
        print("folder created to save prediction file: {0}".format(folder_name))
    pred_file_address = os.path.join(folder_name, file_name)
    pred_file_address += ".pred"
    return pred_file_address


def apply_moses_tokenizer(opt):
    if os.path.exists(opt.moses_tok_script):
        new_src_file_address = opt.src + '.tok'
        basename = os.path.basename(new_src_file_address)
        new_src_file_address = os.path.join(
                                    os.path.join(opt.dir, 'data'),
                                    basename
                                )
        command = "perl {0} -q -l {1} -threads 8 < {2} > {3}".\
            format(opt.moses_tok_script,
                   opt.src_lang,
                   opt.src,
                   new_src_file_address)
        if opt.verbose:
            print("\nMoses tokenization :", command)
        subprocess.check_output(command, shell=True).decode("utf-8")
        opt.src = new_src_file_address
    return opt


def apply_bpe_preprocessing(opt):
    if os.path.exists(opt.bpe_vocab_address) and \
            os.path.exists(opt.bpe_apply_script):
        new_src_file_address = opt.src + '.bpe'
        basename = os.path.basename(new_src_file_address)
        new_src_file_address = os.path.join(
            os.path.join(opt.dir, 'data'),
            basename
        )
        command = "python {0} -c {1} < {2} > {3}". \
            format(opt.bpe_apply_script,
                   opt.bpe_vocab_address,
                   opt.src,
                   new_src_file_address)
        if opt.verbose:
            print("\nApplyting BPE :", command)
        subprocess.check_output(command, shell=True).decode("utf-8")
        opt.src = new_src_file_address
    return opt


def onmt_translate(opt, address, output_address):
    command = "python {0} -model {1} -src {2} -output {3} {4}". \
        format(opt.infer_script,
               address,
               opt.src,
               output_address,
               opt.infer_param)
    if opt.verbose:
        print("\nTranslating :", command)
    subprocess.check_output(command, shell=True).decode("utf-8")


def apply_bpe_postprocessing(opt, output_address):
    if os.path.exists(opt.bpe_vocab_address) and \
            os.path.exists(opt.bpe_apply_script):
        command = "cat {0} | sed -r 's/(@@ )|(@@ ?$)//g' > {1}". \
            format(output_address,
                   output_address + ".bpe")
        if opt.verbose:
            print("\nBPE post-processing :", command)
        subprocess.check_output(command, shell=True).decode("utf-8")
        output_address = output_address + ".bpe"
    return output_address


def apply_moses_detokenizer(opt, output_address):
    if os.path.exists(opt.moses_detok_script):
        command = "perl {0} -q -l {1} -threads 8 < {2} > {3}". \
            format(opt.moses_detok_script,
                   opt.tgt_lang,
                   output_address,
                   output_address + '.detok')
        if opt.verbose:
            print("\nMoses detokenization :", command)
        subprocess.check_output(command, shell=True).decode("utf-8")
        output_address = output_address + '.detok'
    return output_address


def calculate_bleu(opt, output_address):
    ret = ''
    if os.path.exists(opt.blue_score_script) and \
            os.path.exists(opt.tgt):
        command = "perl {0} {1} < {2}". \
            format(opt.blue_score_script,
                   opt.tgt,
                   output_address)
        if opt.verbose:
            print("\nBleu calculate :", command)
        ret = subprocess.check_output(command, shell=True).decode("utf-8")
    return ret


def show_bleu(_str):
    return float(_str.split(",")[0].split()[2])


def main():
    parser = argparse.ArgumentParser(
        description="Accumulate opennmt models and calculate blue score and create charts.",
        prog=((sys.argv[2] + '.py') if os.path.basename(sys.argv[0]) == 'pydoc' else sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    postprocess_opts(parser)
    opt = parser.parse_args()
    assert_address(opt)
    checkpoint_list = retrieve_model_list(opt)
    checkpoint_list = select_checkpoint(checkpoint_list, opt)
    opt = apply_moses_tokenizer(opt)
    opt = apply_bpe_preprocessing(opt)

    for idx, checkpoint in enumerate(checkpoint_list):
        output_address = create_address(checkpoint.address, opt)

        onmt_translate(opt, checkpoint.address, output_address)
        output_address = apply_bpe_postprocessing(opt, output_address)
        output_address = apply_moses_detokenizer(opt, output_address)
        bleu_line = calculate_bleu(opt, output_address)
        if bleu_line == '':
            continue
        checkpoint_list[idx].bleu = bleu_line
        print(show_bleu(checkpoint_list[idx].bleu))

    data_folder = os.path.join(opt.dir, 'data/objs')
    with open(data_folder, 'wb') as file_ptr:
        pickle.dump(checkpoint_list, file_ptr)


if __name__ == "__main__":
    main()
    
    
"""

                               -src "../onmt-runs/${NAME}/data/test.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}" \
                               -tgt "../onmt-runs/${NAME}/data/test.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}" \

SRC_LANG=en
TGT_LANG=ms
NAME="transformer.bpe.${SRC_LANG}-${TGT_LANG}"
bpe="yes"
bpe_arg=''

if [ ${bpe}=="yes" ]; then
    bpe_apply_script="apply_bpe.py"
    bpe_vocab_address="../../dataset-gen/all_dataset/alt_amara_GNOME_KDE4_OpenSubtitles2016_OpenSubtitles2018_Ubuntu/bpe.50000"
    bpe_arg=" -bpe_apply_script ${bpe_apply_script} -bpe_vocab_address ${bpe_vocab_address} "
fi
                               # -select_max_acc \
                               # -select_min_ppl \
cmd="python retrieve_result.py -infer_script "../translate.py" \
                                -src "../../dataset-gen/all_dataset/alt_amara_GNOME_KDE4_OpenSubtitles2016_OpenSubtitles2018_Ubuntu/dummy.en-ms.en" \
                                -tgt "../../dataset-gen/all_dataset/alt_amara_GNOME_KDE4_OpenSubtitles2016_OpenSubtitles2018_Ubuntu/dummy.en-ms.ms" \
                               -infer_param \" -verbose -beam_size 10 \" \
                               -dir "../onmt-runs/${NAME}" \
                               -verbose \
                               -moses_tok_script "tokenizer.perl" \
                               -moses_detok_script "detokenizer.perl" \
                               -blue_score_script "multi-bleu.perl" \
                               -src_lang "${SRC_LANG}" \
                               -tgt_lang "${TGT_LANG}" \
                               ${bpe_arg}  \
                               -chart_folder "../onmt-runs/${NAME}/chart/" \
                               -chart_name ${NAME} \
                               -name ${NAME}"                          
"""