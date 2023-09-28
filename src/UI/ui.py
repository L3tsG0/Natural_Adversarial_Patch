import tkinter as tk
from tkinter import filedialog, DoubleVar
import tkinter.ttk as ttk
from tkinterdnd2 import *
import threading
import time
from UI.AEengine_interface import AEengine_interface

WIDTH=1920
HEIGHT=900


class UI:
    def export_patch(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.interface.export_AEPatch(filename.name)

    def export_model(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.interface.export_AEmodel(filename.name)

    def drop(self, event):
        print(event.data)
        self.text.set(event.data)
        
    def start_learning(self):
        target_of_attack_path: str = self.text.get()
        if not target_of_attack_path:
            return
        try:
            learning_cycles: int = int(self.learning_cycles.get())
        except ValueError as e:
            # self.learning_cyclesがintじゃないとValueErrorが発生する
            return
        if not learning_cycles:
            return
        self.interface.start_learning(target_of_attack_path, learning_cycles)

    def stop_learning(self):
        self.interface.stop_learning()

    # 学習が1エポック回ったタイミングでプレビューする画像を差し替える。インターフェース側から呼ばれる
    def end_1epoch(self, patch_filename: str, attack_success_rate: float, convergence: float):
        self.canvas.delete("all") # いる？
        img = tk.PhotoImage(file=patch_filename, height=self.canvas_height, width=self.canvas_width)
        self.canvas.create_image(self.canvas_width/2, self.canvas_height/2, image=img)
        
        self.attack_success_rate.set(f'攻撃成功率: {attack_success_rate}%')
        self.convergence.set(convergence)

    def search_attack_model_path(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.text.set(filename.name)


    def loop(self):
        prevtime = time.time()   
        while True:
            curtime = time.time()
            self.convergence.set(self.convergence.get() + (curtime-prevtime)*10)
            if self.convergence.get() > 100:
                    self.convergence.set(0)
            prevtime = curtime
            print(self.convergence.get())

    def __init__(self, interface: AEengine_interface):
        self.interface = interface

        self.root = TkinterDnD.Tk()
        self.root.title("demo_Tkinter")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, padx=20, pady=10)


        self.text = tk.StringVar()
        #text.set("Please file drop")

        #entry = ttk.Entry(self.root, textvariable=text)
        #entry.pack(expand=True, fill=tk.X, padx=16, pady=8)

        #entry.drop_target_register(DND_FILES)
        #entry.dnd_bind("<<Drop>>", drop)


        lbl = tk.Label(text='攻撃対象モデル')
        lbl.place(x=WIDTH/100, y=HEIGHT/12)

        # テキストボックス
        txt = ttk.Entry(self.root, textvariable=self.text, width=30)
        self.text.set("Please file drop")
        txt.place(x=WIDTH/100+100, y=HEIGHT/12)
        txt.drop_target_register(DND_FILES)
        txt.dnd_bind("<<Drop>>", self.drop)


        lbl2 = tk.Label(text='学習回数')
        lbl2.place(x=WIDTH/4, y=HEIGHT/12)

        # テキストボックス
        self.learning_cycles = tk.StringVar()
        self.learning_cycles.set("0")
        txt2 = tk.Entry(width=10, textvariable=self.learning_cycles)
        txt2.place(x=WIDTH/4+60, y=HEIGHT/12)
        

        #プレビュー
        self.canvas_width=WIDTH*10//9
        self.canvas_height=HEIGHT*2//3
        self.canvas = tk.Canvas(self.root, bg="white", height=self.canvas_height, width=self.canvas_width)
        self.canvas.place(x=WIDTH/100, y=HEIGHT/8)
        img = tk.PhotoImage(file="image.png", height=self.canvas_height, width=self.canvas_width)
        self.canvas.create_image(self.canvas_width/2, self.canvas_height/2, image=img)

        button1=tk.Button(self.root,text="パッチのエクスポート",width=WIDTH//40,height=HEIGHT//500)
        button1.place(x=WIDTH/20,y=HEIGHT-(HEIGHT/12))
        button1["command"] = self.export_patch
        #button1.pack()

        button2=tk.Button(self.root,text="モデルのエクスポート",width=WIDTH//40,height=HEIGHT//500)
        button2.place(x=WIDTH-(WIDTH/3),y=HEIGHT-(HEIGHT/12))
        button2["command"] = self.export_model
        #button2.pack()

        button3=tk.Button(self.root,text="search",width=5,height=1)
        button3.place(x=WIDTH/100+370,y=HEIGHT/12-5)
        button3["command"] = self.search_attack_model_path
        #button3.pack()

        lbl = tk.Label(text='学習開始')
        lbl.place(x=WIDTH/2.5, y=HEIGHT/12)

        button4=tk.Button(self.root,text="Start",width=5,height=1)
        button4.place(x=WIDTH/2.5+60,y=HEIGHT/12-5)
        button4["command"] = self.start_learning
        #button4.pack()

        lbl = tk.Label(text='学習停止')
        lbl.place(x=WIDTH/2, y=HEIGHT/12)

        button5=tk.Button(self.root,text="stop",width=5,height=1)
        button5.place(x=WIDTH/2+60,y=HEIGHT/12-5)
        button5["command"] = self.stop_learning
        #button5.pack()

        # 攻撃成功率
        self.attack_success_rate = tk.StringVar()
        self.attack_success_rate.set(f'攻撃成功率: 0%')
        lbl_attack_success_rate = tk.Label(textvariable=self.attack_success_rate)
        lbl_attack_success_rate.place(x=WIDTH/10, y=HEIGHT*8/10)
        lbl_attack_success_rate.setvar


        # 学習収束率
        lbl_convergence = tk.Label(text='学習収束率')
        lbl_convergence.place(x=WIDTH/2, y=HEIGHT*8/10)

        self.convergence: DoubleVar = DoubleVar(value=0) # 収束度を表す変数、これの値が変わることでプログレスバーが動くようになる
        self.convergence_progress = ttk.Progressbar(self.root, orient="horizontal", length=WIDTH/4, maximum=100, variable=self.convergence, mode='determinate')
        self.convergence_progress.place(x=WIDTH/2+100, y=HEIGHT*8/10)

        #self.root.mainloop()

    def show(self):
        #thread1 = threading.Thread(target=self.loop)
        #thread1.start()
        self.root.mainloop()