import abc

class AEengine_interface(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    #現時点での学習収束率を返す関数。どう取得できるかがよくわからないのでとりあえずfloatで定義
    #@abc.abstractmethod
    #def get_leaning_convergence_rate(self) -> float:
    #    pass

    #現時点での攻撃成功率を返す関数
    #@abc.abstractmethod
    #def get_attack_success_rate(self) -> float:
    #    pass

    def set_ui_func(self, end_1epoch, set_convergence):
        self.ui_end_1epoch = end_1epoch
        self.set_convergence = set_convergence
        

    @abc.abstractmethod
    def export_AEPatch(self, filename: str):
        pass

    @abc.abstractmethod
    def export_AEmodel(self, filename: str):
        pass

    #学習開始の関数:攻撃対象のパスと学習回数を引数で渡す
    @abc.abstractmethod
    def start_learning(self, Target_model_path: str, Target_image_path: str, patch_image_path: str, learning_cycles: int):
        pass

    #学習停止の関数
    @abc.abstractmethod
    def stop_learning(self):
        pass

    #ラベルの設定
    @abc.abstractmethod
    def set_label(self, setting: bool, csv_path: str):
        pass


    # 生成されたパッチの画像ファイルをAEエンジンから呼び出すことで、UIのプレビューに反映される
    # 同時に攻撃成功率、学習収束率も反映するため、引数で渡す
    #def ui_end_1epoch(self, patch_filename: str, attack_success_rate: float):
    #    self.ui_end_1epoch(patch_filename, attack_success_rate)

    

    