from src.UI.interface import AEengine_interface
from src.UI.ui import UI
import time
import threading
import src.model.attacker as attacker


class AEEngine_interface_implemented(AEengine_interface):
    def export_AEPatch(self, filename: str):
        print(f"Dumped patch: {filename}")

    def export_AEmodel(self, filename: str):
        print(f"Dumped model: {filename}")

    #学習開始の関数:攻撃対象のパスと学習回数を引数で渡す
    def start_learning(self, Target_model_path: str, Target_image_path: str, patch_image_path: str, learning_cycles: int):
        print(f"Start learning.\nmodel path: {Target_model_path}\ntarget image: {Target_image_path}\nlearning cycles: {learning_cycles}")
        attack_thread = threading.Thread(target=attacker.main, args=(Target_model_path, Target_image_path, patch_image_path, learning_cycles))
        #attacker.main(Target_model_path, Target_image_path, patch_image_path, learning_cycles)
        attack_thread.start()

    #学習停止の関数
    def stop_learning(self):
        print(f"Stop learning.")
        attacker.stop_cycles()

    def set_label(self, setting: bool, csv_path: str):
        attacker.set_usecsv(setting, csv_path)
        

if __name__ == "__main__":
    interface : AEengine_interface = AEEngine_interface_implemented()
    ui: UI = UI(interface)
    interface.set_ui_func(ui.end_1epoch, ui.set_convergence)
    attacker.set_interface(interface)
    ui.show()