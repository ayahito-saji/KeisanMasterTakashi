import argparse
import random
import os

class ExpressionCreator():
    def __init__(self, max_digit=1, max_depth=1, include_kakezan=False, include_warizan=False):
        self.max_digit = max_digit
        self.max_depth = max_depth
        self.include_kakezan = include_kakezan
        self.include_warizan = include_warizan
    
    def create(self):
        while True:
            expr, answer = self.__expression(1, None)
            if eval(expr) == answer:
                return expr, answer
            else:
                raise Exception(f"{expr} != {answer}")
            if answer <= 10 ** (max_degit * 2) - 1:
                break
        return expr, answer
    
    def __expression(self, current_depth, parent_operator):
        operators = ["+", "-"]
        if self.include_kakezan:
            operators.append("*")
        if self.include_warizan:
            operators.append("/")

        operator = random.choice(operators)

        if operator == "+":
            if current_depth < self.max_depth and random.random() < 0.5:
                expr0, answer0 = self.__expression(current_depth+1, "+")
            else:
                expr0, answer0 = self.__primary()

            if current_depth < self.max_depth and random.random() < 0.5:
                expr1, answer1 = self.__expression(current_depth+1, "+")
            else:
                expr1, answer1 = self.__primary()

            if parent_operator is "*" or parent_operator is "/":
                return "("+expr0 + "+" + expr1+")", answer0 + answer1
            else:
                return expr0 + "+" + expr1, answer0 + answer1

        elif operator == "-":
            while True:
                if current_depth < self.max_depth and random.random() < 0.5:
                    expr0, answer0 = self.__expression(current_depth+1, "-")
                else:
                    expr0, answer0 = self.__primary()
                
                if current_depth < self.max_depth and random.random() < 0.5:
                    expr1, answer1 = self.__expression(current_depth+1, "-")
                    expr1 = "(" + expr1 + ")"
                else:
                    expr1, answer1 = self.__primary()
                
                if answer0 - answer1 >= 1:
                    break

            if parent_operator is "*" or parent_operator is "/":
                return "("+expr0 + "-" + expr1+")", answer0 - answer1
            else:
                return expr0 + "-" + expr1, answer0 - answer1

        elif operator == "*":
            if current_depth < self.max_depth and random.random() < 0.5:
                expr0, answer0 = self.__expression(current_depth+1, "*")
            else:
                expr0, answer0 = self.__primary()

            if current_depth < self.max_depth and random.random() < 0.5:
                expr1, answer1 = self.__expression(current_depth+1, "*")
            else:
                expr1, answer1 = self.__primary()

            return expr0 + "*" + expr1, answer0 * answer1

        elif operator == "/":
            while True:
                if current_depth < self.max_depth and random.random() < 0.5:
                    expr0, answer0 = self.__expression(current_depth+1, "/")
                else:
                    expr0, answer0 = self.__primary()
                
                if current_depth < self.max_depth and random.random() < 0.5:
                    expr1, answer1 = self.__expression(current_depth+1, None)
                    expr1 = "(" + expr1 + ")"
                else:
                    expr1, answer1 = self.__primary()
                
                if answer0 % answer1 == 0:
                    break

            return expr0 + "/" + expr1, int(answer0 / answer1)


    def __primary(self):
        """
        max_digit: 式に登場する最大の桁数
        return ランダムな値
        """
        v = random.randint(1, 10**self.max_digit-1)
        return str(v), v

def create_dataset(dir_path,
                   prefix,
                   max_digit,
                   max_depth,
                   number_of_train_dataset,
                   number_of_test_dataset,
                   include_kakezan,
                   include_warizan
                   ):
    expression_creator = ExpressionCreator(max_digit, max_depth, include_kakezan, include_warizan)

    if prefix != "":
        train_file_name = f"{prefix}_train"
        test_file_name = f"{prefix}_test"
    else:
        train_file_name = "train"
        test_file_name = "test"

    with open(os.path.join(dir_path, train_file_name+".txt"), "w") as fp:
        for i in range(number_of_train_dataset):
            expression, answer = expression_creator.create()
            fp.write(f"{expression}\t{answer}\n")

    with open(os.path.join(dir_path, test_file_name+".txt"), "w") as fp:
        for i in range(number_of_test_dataset):
            expression, answer = expression_creator.create()
            fp.write(f"{expression}\t{answer}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算数データセットを作成します')
    parser.add_argument("--dir_path",
                        default="../data/datasets",
                        help="データセットの保存先のディレクトリのパス")

    parser.add_argument("-p", "--prefix",
                        default="",
                        help="データセットの名前にprefixをつけられます")

    parser.add_argument("--max_digit",
                        default=1,
                        type=int,
                        help="データセットの式が含む数字の最大の桁数")

    parser.add_argument("--max_depth",
                        default=2,
                        type=int,
                        help="データセットの式が含む最大の括弧の深さ")

    parser.add_argument("--number_of_train_dataset",
                        default=10000,
                        type=int,
                        help="訓練データセットの個数")

    parser.add_argument("--number_of_test_dataset",
                        default=1000,
                        type=int,
                        help="テストデータセットの個数")

    parser.add_argument("-k", "--include_kakezan",
                        action='store_true',
                        help="掛け算を含むかどうか")

    parser.add_argument("-w", "--include_warizan",
                        action='store_true',
                        help="割り算を含むかどうか")
    args = parser.parse_args()
    create_dataset(
        dir_path=args.dir_path,
        prefix=args.prefix,
        max_digit=args.max_digit,
        max_depth=args.max_depth,
        number_of_train_dataset=args.number_of_train_dataset,
        number_of_test_dataset=args.number_of_test_dataset,
        include_kakezan=args.include_kakezan,
        include_warizan=args.include_warizan
    )