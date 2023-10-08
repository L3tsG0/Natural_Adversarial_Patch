from src.UI.interface import AEengine_interface
from src.UI.ui import UI
import time
import threading
import src.model.attacker as attacker
import src.model.generate_moth as gan
import os

def _start_learning(Target_model_path: str, Target_image_path: str, patch_image_path: str, learning_cycles: int, iteration_cycles: int, useGAN: bool):
    if useGAN is True: 
        gan.main(patch_image_path, learning_cycles)
        patch_image_path = "export/GAN_generated_images/image_0.png"
    attacker.main(Target_model_path, Target_image_path, patch_image_path, iteration_cycles)


class AEEngine_interface_implemented(AEengine_interface):
    def __init__(self) -> None:
        self.attack = None
        super().__init__()

    def export_AEPatch(self, filename: str):
        print(f"Dumped patch: {filename}")

    def export_AEmodel(self, filename: str):
        print(f"Dumped model: {filename}")

    #学習開始の関数:攻撃対象のパスと学習回数を引数で渡す
    def start_learning(self, Target_model_path: str, Target_image_path: str, patch_image_path: str, learning_cycles: int, iteration_cycles: int, useGAN: bool):
        if self.attack:
            if self.attack.is_alive():
                return
            else:
                self.attack.join()
        attacker.stop = False
        gan.stop = False
        print(f"Start learning.\nmodel path: {Target_model_path}\ntarget image: {Target_image_path}\nlearning cycles: {learning_cycles}\niteration cycles: {iteration_cycles}\nuseGAN: {useGAN}")
        self.attack = threading.Thread(target=_start_learning, args=(Target_model_path, Target_image_path, patch_image_path, learning_cycles, iteration_cycles, useGAN))
        self.attack.start()
        # if useGAN is True: 
        #     gan_thread = 
        # else:
        #     attacker.main(Target_model_path, Target_image_path, patch_image_path, learning_cycles)
        #     attack_thread = threading.Thread(target=attacker.main, args=(Target_model_path, Target_image_path, patch_image_path, iteration_cycles))
        #     attack_thread.start()
        

    #学習停止の関数
    def stop_learning(self):
        print(f"Stop learning.")
        attacker.stop_cycles()
        gan.stop_cycles()

    def set_label(self, setting: bool, csv_path: str):
        attacker.set_usecsv(setting, csv_path)
        

if __name__ == "__main__":
    interface : AEengine_interface = AEEngine_interface_implemented()
    ui: UI = UI(interface)
    interface.set_ui_func(ui.end_1epoch, ui.set_convergence, ui.end_1optimize_iteration)
    attacker.set_interface(interface)
    gan.set_interface(interface)
    ui.show()