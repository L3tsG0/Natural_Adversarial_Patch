import abc

class AEengine_interface(metaclass=abc.ABCMeta):
    def __init__(self, replace_func) -> None:
        self.replace_func = replace_func

    #現時点での学習収束率を返す関数。どう取得できるかがよくわからないのでとりあえずfloatで定義
    @abc.abstractmethod
    def get_leaning_convergence_rate(self) -> float:
        pass

    #現時点での攻撃成功率を返す関数
    @abc.abstractmethod
    def get_attack_success_rate(self) -> float:
        pass

    # 生成されたパッチの画像ファイルをAEエンジンから呼び出すことで、UIのプレビューに反映される
    def replace_preview(self, filename: str):
        self.replace_func(filename)