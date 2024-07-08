import argparse


def remove_oov_in_lexicon(units_path: str, lexicon_path: str, new_lexicon_path: str,
                          special_word: set = "{'<blank>', '<unk>', '<sos/eos>'}"):
    """ 删除lexicon.txt中含 ”超纲字“ 的行，“超纲字”指units.txt中没有的字

            Args:
                units_path (str): units.txt的路径
                lexicon_path (str): lexicon.txt的路径
                new_lexicon_path (str): 产生的new_lexicon_path.txt的路径
                special_word (str): units.txt中的特殊字，包括<blank>、<unk>、<sos/eos>等

            Returns:
                None
        """
    units_set = set()
    with open(units_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            word = line.split(' ')[0]
            if word not in special_word:
                units_set.add(word)

    with open(lexicon_path, 'r', encoding='utf8') as fr, open(new_lexicon_path, 'w', encoding='utf8') as fw:
        for line in fr.readlines():
            list_line = list(line.split(' ')[0])
            is_oov_line = False
            for c in list_line:
                if c not in units_set:
                    is_oov_line = True
            if not is_oov_line:
                fw.write(line)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--units_path', default='units.txt', help='units_path')
parser.add_argument('--lexicon_path', default='lexicon.txt', help='lexicon_path')
parser.add_argument('--new_lexicon_path', default='new_lexicon.txt', help='new_lexicon_path')
args = parser.parse_args()
remove_oov_in_lexicon(args.units_path, args.lexicon_path, args.new_lexicon_path)
