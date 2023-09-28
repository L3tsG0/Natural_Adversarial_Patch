from UI.AEengine_interface import AEengine_interface
from UI.ui import UI
import time
import threading

def loop(interface: AEengine_interface):
    prevtime = time.time()   
    while True:
        curtime = time.time()
        if curtime - prevtime > 5:
            interface.end_1epoch("image.png", 59.6, 60)
            return


class AEEngine(AEengine_interface):
    def export_AEPatch(self, filename: str):
        print(f"Dumped patch: {filename}")

    def export_AEmodel(self, filename: str):
        print(f"Dumped model: {filename}")

    #学習開始の関数:攻撃対象のパスと学習回数を引数で渡す
    def start_learning(self, Target_of_attack_path: str, learning_cycles: int):
        print(f"Start learning.\npath: {Target_of_attack_path}\nlearning cycles: {learning_cycles}")

    #学習停止の関数
    def stop_learning(self):
        print(f"Stop learning:")

if __name__ == "__main__":
    interface : AEengine_interface = AEEngine()
    ui: UI = UI(interface)
    interface.set_end_1epoch(ui.end_1epoch)
    thread1 = threading.Thread(target=loop, args=[interface])
    thread1.start()
    ui.show()