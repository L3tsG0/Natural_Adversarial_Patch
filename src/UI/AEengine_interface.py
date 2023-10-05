import abc

class AEengine_interface(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.ui_end_1epoch = None

    #現時点での学習収束率を返す関数。どう取得できるかがよくわからないのでとりあえずfloatで定義
    #@abc.abstractmethod
    #def get_leaning_convergence_rate(self) -> float:
    #    pass

    #現時点での攻撃成功率を返す関数
    #@abc.abstractmethod
    #def get_attack_success_rate(self) -> float:
    #    pass

    def set_end_1epoch(self, ui_end_1epoch):
        self.ui_end_1epoch = ui_end_1epoch

    @abc.abstractmethod
    def export_AEPatch(self, filename: str):
        pass

    @abc.abstractmethod
    def export_AEmodel(self, filename: str):
        pass

    #学習開始の関数:攻撃対象のパスと学習回数を引数で渡す
    @abc.abstractmethod
    def start_learning(self, Target_of_attack_path: str, learning_cycles: int):
        pass

    #学習停止の関数
    @abc.abstractmethod
    def stop_learning(self):
        pass
        
    # 生成されたパッチの画像ファイルをAEエンジンから呼び出すことで、UIのプレビューに反映される
    # 同時に攻撃成功率、学習収束率も反映するため、引数で渡す
    def end_1epoch(self, patch_filename: str, attack_success_rate: float, convergence: float):
        self.ui_end_1epoch(patch_filename, attack_success_rate, convergence)

    

    