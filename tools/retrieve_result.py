import os
import argparse
import sys
import subprocess

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
                       required=True,
                       help="The address of the source file that will be translated by model."
                            "Each line contains a single example.")
    group.add_argument("-tgt",
                       type=str,
                       required=True,
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
    group.add_argument("-select_max",
                        action='store_true',
                        help="Select one the model based on best acc, ppl within the -dir folder.")
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
    group.add_argument("-blue_score_script",
                        type=str,
                        required=True,
                        help="The address of the script that will calculate the blue score."
                             "tokenizer option: multi-bleu-detok.perl, multi-bleu.perl")
    group.add_argument("-bpe_process",
                        action='store_true',
                        help="If BPE process is applied in dataset preprocessing, "
                             "it will do the postprocessing related to BPE.")

    group = parser.add_argument_group('Chart')
    group.add_argument("-chart",
                        type=str,
                        default="ms",
                        help="short identifier of the target language. ex: ms")
    group.add_argument("-chart_folder",
                        type=str,
                        default="ms",
                        help="short identifier of the target language. ex: ms")
    group.add_argument("-chart_name",
                        type=str,
                        default="ms",
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
    files = os.listdir(opt.dir)
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
                    os.path.join(opt.dir, address),
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
    if opt.select_max:
        mx_idx = mx_acc = mx_ppl = 0
        for (idx, chk_pt) in enumerate(checkpoint_list):
            if chk_pt.acc > mx_acc:
                mx_idx = idx
                mx_acc = chk_pt.acc
                mx_ppl = chk_pt.ppl
            if chk_pt.acc == mx_acc and \
                chk_pt.ppl > mx_ppl:
                mx_idx = idx
                mx_acc = chk_pt.acc
                mx_ppl = chk_pt.ppl
        return checkpoint_list[mx_idx:mx_idx+1]

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
    if opt.verbose:
        print("folder created to save prediction file: {0}".format(folder_name))
    os.makedirs(folder_name, exist_ok=True)
    pred_file_address = os.path.join(folder_name, file_name)
    pred_file_address += ".pred"
    return pred_file_address


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
    for checkpoint in checkpoint_list:
        output_address = create_address(checkpoint.address, opt)
        command = "python {0} -model {1} -src {2} -output {3} {4}".\
            format(opt.infer_script,
                   checkpoint.address,
                   opt.src,
                   output_address,
                   opt.infer_param)
        if opt.verbose:
            print("Running :", command)
        res = subprocess.check_output(command, shell=True).decode("utf-8")
        # msg = ">> " + res.strip()
        # cat $OUT / test / test.out.bpe | sed - E 's/(@@ )|(@@ ?$)//g' > $OUT / test / test.out
        command = "cat {0} | sed -r 's/(@@ )|(@@ ?$)//g' > {1}".format(output_address, output_address+".bpe")
        exit(1)


if __name__ == "__main__":
    main()