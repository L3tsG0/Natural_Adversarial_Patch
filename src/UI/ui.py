import tkinter as tk
from tkinter import filedialog, DoubleVar
import tkinter.ttk as ttk
from tkinterdnd2 import *
import threading
import time
from src.UI.interface import AEengine_interface
from PIL import Image, ImageTk

WIDTH=1920
HEIGHT=900


class UI:
    
    def start_learning(self):
        target_model_path: str = self.model_path.get()
        target_image_path: str = self.target_image_textbox.get()
        patch_path: str = self.patch_textbox.get()
        csv_path: str = self.csv_textbox.get()
        if not target_model_path or not target_image_path or not patch_path:
            return
        if self.radio_value.get() == 1:
            if not csv_path:
                return
        self.interface.set_label(True if self.radio_value.get() == 1 else False, csv_path)

        try:
            learning_cycles: int = int(self.learning_cycles.get())
        except ValueError as e:
            # self.learning_cyclesがintじゃないとValueErrorが発生する
            return
        if not learning_cycles:
            return
        self.interface.start_learning(target_model_path, target_image_path, patch_path, learning_cycles)

    def stop_learning(self):
        self.interface.stop_learning()

    # 学習が1エポック回ったタイミングでプレビューする画像を差し替える。インターフェース側から呼ばれる
    def end_1epoch(self, patch_filename: str, attack_success_rate: float):
        self.canvas.delete("all") # いる？

        img = Image.open(patch_filename).resize((self.canvas_height, self.canvas_height))
        #self.img = tk.PhotoImage(file=patch_filename, height=self.canvas_height, width=self.canvas_width)
        self.img = ImageTk.PhotoImage(img)
        self.canvas.create_image(self.canvas_width/2, self.canvas_height/2, image=self.img)
        
        self.attack_success_rate.set(f'最大報酬: {attack_success_rate}')
        #self.convergence.set(convergence)

    def set_convergence(self, convergence: float):
        self.convergence.set(convergence)

    

    def __init__(self, interface: AEengine_interface):
        self.interface = interface

        self.root = TkinterDnD.Tk()
        self.root.title("demo_Tkinter")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")

        #frame = ttk.Frame(self.root)
        #frame.pack(fill=tk.BOTH, padx=20, pady=10)


        self.model_path = tk.StringVar()
        #text.set("Please file drop")

        #entry = ttk.Entry(self.root, textvariable=text)
        #entry.pack(expand=True, fill=tk.X, padx=16, pady=8)

        #entry.drop_target_register(DND_FILES)
        #entry.dnd_bind("<<Drop>>", drop)

        # 攻撃対象モデル
        lbl = tk.Label(text='攻撃対象モデル')
        lbl.place(x=WIDTH/100, y=HEIGHT/15)
        txt = ttk.Entry(self.root, textvariable=self.model_path, width=30)
        self.model_path.set("D&D the target model file (.pth)")
        txt.place(x=WIDTH/100+100, y=HEIGHT/15)
        txt.drop_target_register(DND_FILES)
        txt.dnd_bind("<<Drop>>", self.__drop_model)

        # 学習回数
        lbl2 = tk.Label(text='学習回数')
        lbl2.place(x=WIDTH/4, y=HEIGHT/15)
        self.learning_cycles = tk.StringVar()
        self.learning_cycles.set("0")
        txt2 = tk.Entry(self.root, width=10, textvariable=self.learning_cycles)
        txt2.place(x=WIDTH/4+60, y=HEIGHT/15)
        
        #パッチの画像のパス
        lbl_patch = tk.Label(text='パッチに使う画像のパス')
        lbl_patch.place(x=WIDTH/100, y=HEIGHT/15+30)
        self.patch_textbox = tk.StringVar()
        entry = tk.Entry(self.root, width=30, textvariable=self.patch_textbox)
        entry.place(x=WIDTH/100+150, y=HEIGHT/15+30)
        entry.drop_target_register(DND_FILES)
        entry.dnd_bind("<<Drop>>", self.__drop_patch)

        button=tk.Button(self.root,text="search",width=5,height=1)
        button.place(x=WIDTH/100+400,y=HEIGHT/15+30-5)
        button["command"] = self.__search_patch_path

        #攻撃対象の画像のパス
        lbl_target_image = tk.Label(text='攻撃対象の画像のパス')
        lbl_target_image.place(x=WIDTH/100, y=HEIGHT/15+60)
        self.target_image_textbox = tk.StringVar()
        entry = tk.Entry(self.root, width=30, textvariable=self.target_image_textbox)
        entry.place(x=WIDTH/100+150, y=HEIGHT/15+60)
        entry.drop_target_register(DND_FILES)
        entry.dnd_bind("<<Drop>>", self.__drop_target)
        
        button=tk.Button(self.root,text="search",width=5,height=1)
        button.place(x=WIDTH/100+400,y=HEIGHT/15+60-5)
        button["command"] = self.__search_target_image_path
        
        
        #ラジオボタン
        self.radio_value = tk.IntVar(value = 0) #初期値設定あり
        #self.radio_value = tk.IntVar()         #初期値設定なし

        radio0 = tk.Radiobutton(self.root,
                                text = "正解ラベルをモデルの出力を基に与える",
                                variable = self.radio_value,
                                value = 0)
        radio0.place(x=WIDTH*2/3 , y=HEIGHT/15)
        radio1 = tk.Radiobutton(self.root,
                                text = "正解ラベルを人が与える",
                                variable = self.radio_value,
                                value = 1)
        radio1.place(x=WIDTH*2/3 ,y=HEIGHT/15+20)

        # ラベル用CSVのパスを入力するフィールド
        lbl_csv = tk.Label(text='ラベル(.csv)')
        lbl_csv.place(x=WIDTH*2/3, y=HEIGHT/15+50)
        self.csv_textbox = tk.StringVar()
        entry = tk.Entry(self.root, width=30, textvariable=self.csv_textbox)
        entry.place(x=WIDTH*2/3+100, y=HEIGHT/15+50)
        entry.drop_target_register(DND_FILES)
        entry.dnd_bind("<<Drop>>", self.__drop_csv)
        
        button=tk.Button(self.root,text="search",width=5,height=1)
        button.place(x=WIDTH*2/3+350,y=HEIGHT/15+50-5)
        button["command"] = self.__search_csv_path
        

        #プレビュー
        self.canvas_width=WIDTH*10//9
        self.canvas_height=HEIGHT*3//5
        self.canvas = tk.Canvas(self.root, bg="white", height=self.canvas_height, width=self.canvas_width)
        self.canvas.place(x=WIDTH/100, y=HEIGHT/5)
        #self.img = tk.PhotoImage(file="image2.png", height=self.canvas_height, width=self.canvas_width)
        #self.canvas.create_image(self.canvas_width//2, self.canvas_height//2, image=self.img)

        button1=tk.Button(self.root,text="Export Patch",width=WIDTH//40,height=HEIGHT//500)
        button1.place(x=WIDTH/20,y=HEIGHT-(HEIGHT/12))
        button1["command"] = self.__export_patch
        #button1.pack()

        button2=tk.Button(self.root,text="Export AE Model",width=WIDTH//40,height=HEIGHT//500)
        button2.place(x=WIDTH-(WIDTH/3),y=HEIGHT-(HEIGHT/12))
        button2["command"] = self.__export_model
        #button2.pack()

        button3=tk.Button(self.root,text="search",width=5,height=1)
        button3.place(x=WIDTH/100+370,y=HEIGHT/15-5)
        button3["command"] = self.__search_attack_model_path
        #button3.pack()

        lbl = tk.Label(text='学習開始')
        lbl.place(x=WIDTH/2.5, y=HEIGHT/15)

        button4=tk.Button(self.root,text="Start",width=5,height=1)
        button4.place(x=WIDTH/2.5+60,y=HEIGHT/15-5)
        button4["command"] = self.start_learning
        #button4.pack()

        lbl = tk.Label(text='学習停止')
        lbl.place(x=WIDTH/2, y=HEIGHT/15)

        button5=tk.Button(self.root,text="stop",width=5,height=1)
        button5.place(x=WIDTH/2+60,y=HEIGHT/15-5)
        button5["command"] = self.stop_learning
        #button5.pack()

        # 攻撃成功率
        self.attack_success_rate = tk.StringVar()
        self.attack_success_rate.set(f'最大報酬: ')
        lbl_attack_success_rate = tk.Label(textvariable=self.attack_success_rate)
        lbl_attack_success_rate.place(x=WIDTH/10, y=HEIGHT*10/12)
        lbl_attack_success_rate.setvar


        # 学習収束率
        lbl_convergence = tk.Label(text='学習収束率')
        lbl_convergence.place(x=WIDTH/2, y=HEIGHT*10/12)

        self.convergence: DoubleVar = DoubleVar(value=0) # 収束度を表す変数、これの値が変わることでプログレスバーが動くようになる
        self.convergence_progress = ttk.Progressbar(self.root, orient="horizontal", length=WIDTH/4, maximum=1, variable=self.convergence, mode='determinate')
        self.convergence_progress.place(x=WIDTH/2+100, y=HEIGHT*10/12)

        

    def show(self):
        #thread1 = threading.Thread(target=self.loop)
        #thread1.start()
        self.root.mainloop()

        
    def __export_patch(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.interface.export_AEPatch(filename.name)

    def __export_model(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.interface.export_AEmodel(filename.name)

    def __drop_model(self, event):
        print(event.data)
        self.model_path.set(event.data)

    def __drop_patch(self, event):
        print(event.data)
        self.patch_textbox.set(event.data)

    def __drop_target(self, event):
        print(event.data)
        self.target_image_textbox.set(event.data)

    def __drop_csv(self, event):
        print(event.data)
        self.csv_textbox.set(event.data)

    def __search_attack_model_path(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.model_path.set(filename.name)

    def __search_patch_path(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.patch_textbox.set(filename.name)


    def __search_csv_path(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.csv_textbox.set(filename.name)

    def __search_target_image_path(self):
        filename = filedialog.askopenfile()
        if not filename:
            return
        self.target_image_textbox.set(filename.name)

    def __loop(self):
        prevtime = time.time()   
        while True:
            curtime = time.time()
            self.convergence.set(self.convergence.get() + (curtime-prevtime)*10)
            if self.convergence.get() > 100:
                    self.convergence.set(0)
            prevtime = curtime
            print(self.convergence.get())