import tkinter as tk
from tkinter import filedialog, DoubleVar
import tkinter.ttk as ttk
from tkinterdnd2 import *
import threading
import time


WIDTH=1920
HEIGHT=900
def export_patch():
    filename = filedialog.askopenfile()

def export_model():
    filename = filedialog.askopenfile()

def drop(event):
    print(event.data)
    text.set(event.data)

# 学習が1エポック回ったタイミングでプレビューする画像を指し返る。インターフェース側から呼ばれる
def replace_image():
     pass

root = TkinterDnD.Tk()
root.title("demo_Tkinter")
root.geometry(f"{WIDTH}x{HEIGHT}")

frame = ttk.Frame(root)
frame.pack(fill=tk.BOTH, padx=20, pady=10)

# ラベルの作成
label = tk.Label(root, text="This is the Label.")

text = tk.StringVar()
#text.set("Please file drop")

#entry = ttk.Entry(root, textvariable=text)
#entry.pack(expand=True, fill=tk.X, padx=16, pady=8)

#entry.drop_target_register(DND_FILES)
#entry.dnd_bind("<<Drop>>", drop)

#ラベルの表示
label.pack(pady=0)

lbl = tk.Label(text='攻撃対象モデル')
lbl.place(x=WIDTH/100, y=HEIGHT/12)

# テキストボックス
txt = ttk.Entry(root, textvariable=text, width=30)
text.set("Please file drop")
txt.place(x=WIDTH/100+100, y=HEIGHT/12)
txt.drop_target_register(DND_FILES)
txt.dnd_bind("<<Drop>>", drop)



lbl2 = tk.Label(text='学習回数')
lbl2.place(x=WIDTH/4, y=HEIGHT/12)

# テキストボックス
txt2 = tk.Entry(width=10)
txt2.place(x=WIDTH/4+60, y=HEIGHT/12)

#プレビュー
canvas_width=1860
canvas_height=700
canvas = tk.Canvas(root, bg="white", height=canvas_height, width=canvas_width)
canvas.place(x=canvas_width/62, y=canvas_height/6)
img = tk.PhotoImage(file="image.png", height=canvas_height, width=canvas_width)
canvas.create_image(canvas_width/2, canvas_height/2, image=img)

button1=tk.Button(root,text="パッチのエクスポート",width=WIDTH//40,height=HEIGHT//500)
button1.place(x=WIDTH/20,y=HEIGHT-(HEIGHT/12))
button1["command"] = export_patch
#button1.pack()

button2=tk.Button(root,text="モデルのエクスポート",width=WIDTH//40,height=HEIGHT//500)
button2.place(x=WIDTH-(WIDTH/3),y=HEIGHT-(HEIGHT/12))
button2["command"] = export_patch
#button2.pack()

button3=tk.Button(root,text="search",width=5,height=1)
button3.place(x=WIDTH/100+370,y=HEIGHT/12-5)
button3["command"] = export_patch
#button3.pack()

lbl = tk.Label(text='学習開始')
lbl.place(x=WIDTH/2.5, y=HEIGHT/12)

button4=tk.Button(root,text="Start",width=5,height=1)
button4.place(x=WIDTH/2.5+60,y=HEIGHT/12-5)
button4["command"] = export_patch
#button4.pack()

lbl = tk.Label(text='学習停止')
lbl.place(x=WIDTH/2, y=HEIGHT/12)

button5=tk.Button(root,text="stop",width=5,height=1)
button5.place(x=WIDTH/2+60,y=HEIGHT/12-5)
button5["command"] = export_patch
#button5.pack()

# 攻撃成功率

# 学習収束率
lbl_convergence = tk.Label(text='学習収束率')
lbl_convergence.place(x=WIDTH/2, y=HEIGHT*8/10)

convergence: DoubleVar = DoubleVar(value=0) # 収束度を表す変数、これの値が変わることでプログレスバーが動くようになる
convergence_progress = ttk.Progressbar(root, orient="horizontal", length=WIDTH/4, maximum=100, variable=convergence, mode='determinate')
convergence_progress.place(x=WIDTH/2+100, y=HEIGHT*8/10)

def loop():
    prevtime = time.time()   
    while True:
        curtime = time.time()
        convergence.set(convergence.get() + (curtime-prevtime)*10)
        if convergence.get() > 100:
                convergence.set(0)
        prevtime = curtime
        print(convergence.get())

#root.mainloop()
thread1 = threading.Thread(target=loop)
thread1.start()
root.mainloop()