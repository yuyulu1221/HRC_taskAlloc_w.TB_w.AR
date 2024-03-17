
#%% 
import numpy as np
import pandas as pd
import copy

#%% Test
tb_abbr = {
    "TE": "Transport Empty",
    "TL": "Transport Loaded",
    "G": "Grasp",
    "RL": "Release Load",
    "A": "Assemble",
    "DA": "Disassemble",
    "P": "Position",
    "H": "Hold",
}

# analyze_res = {
#     "Type": ["TE", "G", "P", "RL", "TE", "G", "P", "RL"],
#     "From": ["L", "-", "-", "-", "Pillar", "-", "-", "-"],
#     "To": ["Pillar", "-", "-", "-", "BottomPlate", "-", "-", "-"],
#     "Obj1": ["-", "Pillar", "Pillar", "-", "-", "Pillar", "Pillar", "-"],
#     "Obj2": ["-", "-", "BottomPlate", "-", "-", "-", "BottomPlate", "-"]
# }
 
# Pos = {
#     "L": np.array([-0.2, 0, 0.1]),    
#     "R": np.array([ 0.2, 0, 0.1]),
#     "Bot": np.array([0, 0, 0.6]),
#     "Pillar": np.array([-0.3, 0, 0.3]),
#     "Bush": np.array([-0.2, 0, 0.3]),
#     "TopPlate": np.array([-0.2, 0, 0.5]),
#     "BottomPlate": np.array([-0.2, 0, 0.25]),
#     "Done": np.array([0.3, 0, 0.4]),
# }

# Obj = ["Pillar", "Bush", "TopPlate", "BottomPlate"]



#%% 動素
class Therblig(object):
    def __init__(self, Type:str, From:np.array=None, To:np.array=None, Obj1=None, Obj2=None):
        if tb_abbr.get(Type) == None:
            raise ValueError("This type of therblig is not exist")
        self.type = Type
        self.start: float
        self.end: float
        self.From = From
        self.To = To
        self.Obj1 = Obj1
        self.Obj2 = Obj2
        self.time: float
        # self.next = None
        
    def __repr__(self):
        return str(self.type)
    
#%% Inherit class        
# class TE(Therblig):
#     def __init__(self, From, To) -> None:
#         super().__init__()
#         self.From = From
#         self.To = To
#         self.moveSpeed = .8
#         self.time = np.linalg.norm(self.To - self.From) / self.moveSpeed
        
#     def estimated_time(self) -> float:
#         return np.linalg.norm(self.To - self.From) / self.moveSpeed     

# class TL(Therblig):
#     def __init__(self, From, To, Obj) -> None:
#         super().__init__()
#         self.From = From
#         self.To = To
#         self.Obj = Obj
#         self.moveSpeed = .6
#         self.time = np.linalg.norm(self.To - self.From) / self.moveSpeed
        
#     def estimated_time(self) -> float:
#         return np.linalg.norm(self.To - self.From) / self.moveSpeed   
        
# class G(Therblig):
#     def __init__(self, Obj) -> None:
#         super().__init__()
#         self.Obj = Obj
#         self.time = .1

# class RL(Therblig):
#     def __init__(self, Obj) -> None:
#         super().__init__()
#         self.Obj = Obj
#         self.time = .1
        
# class H(Therblig):
#     def __init__(self, Obj) -> None:
#         super().__init__()
#         self.Obj = Obj
#     # TODO How to estimate Holding time??
        
# class A(Therblig):
#     def __init__(self, Obj1, Obj2) -> None:
#         super().__init__()
#         self.Obj1 = Obj1
#         self.Obj2 = Obj2
        
# class DA(Therblig):
#     def __init__(self, Obj1, Obj2) -> None:
#         super().__init__()
#         self.Obj1 = Obj1
#         self.Obj2 = Obj2
        
# class P(Therblig):
#     def __init__(self, Obj1, Obj2) -> None:
#         super().__init__()
#         self.Obj1 = Obj1
#         self.Obj2 = Obj2

#%% 動素序列(Linked-List)
# class TherbligLinkedList(object):
#     def __init__(self):
#         self.head = None
        
#     def __str__(self):
#         ptr = self.head
#         if ptr == None:
#             return "Empty"
#         ret = ptr.type
#         ptr = ptr.next
#         while ptr != None:
#             ret = ret + ", " + ptr.type
#             ptr = ptr.next
#         return ret
        
#     def set_head(self, tb):
#         self.head = tb
        
#     def read(self, df:pd.DataFrame):
#         for idx, row in df.iterrows():
#             therblig = Therblig(row["Type"], Pos[row["From"]], Pos[row["To"]], row["Obj1"], row["Obj2"])
#             if self.head == None:
#                 self.head = copy.copy(therblig)
#                 self.ptr = self.head
#             else:
#                 self.ptr.next = copy.copy(therblig)
#                 self.ptr = self.ptr.next
#%% 動素序列
class Therbligs(object):
    def __init__(self, Pos):
        self.list = []
        self.Pos = Pos
    def __repr__(self):
        return ", ".join(map(str, self.list))
        
    def read(self, df:pd.DataFrame):
        for idx, row in df.iterrows():
            therblig = Therblig(
                Type=row["Type"], 
                From=self.Pos.get(row["From"], None), 
                To=self.Pos.get(row["To"], None), 
                Obj1=row["Obj1"], 
                Obj2=row["Obj2"]
            )
            self.list.append(copy.copy(therblig))


#%% 單手任務
class OHT(object):
    def __init__(self, *args):
        self.list = []
        for arg in args:
            if not isinstance(arg, Therbligs):
                continue
            self.read(arg)
            
    def __repr__(self):
        return ", ".join(map(str, self.list))
            
    def read(self, tbs:Therbligs):
        tmp = []
        for tb in tbs.list:
            if len(tmp) != 0:
                if tmp[-1] == "A" or tmp[-1].type == "DA" and tb.type != "RL":
                    raise ValueError("Invalid therblig list")  
            tmp.append(tb)
            if tb.type == "RL":
                self.list.append(tmp.copy())
                tmp.clear()
                
            # if tmp.count != 0 \
            # and (tmp[-1].type == "A" or tmp[-1].type == "DA") \
            # and tb.type != "RL":
            #     tmp.append(Therblig(Type="RL", Obj1=tmp[-2].Obj1))
                 
    
    
#%% TBHandler
class TBHandler(object):
    def __init__(self):
        self.Pos = {}
        self.tbsl:Therbligs
        self.tbsr:Therbligs
        self.OHT:OHT

    def save_pos(self):
        pos_df = pd.read_excel("data.xlsx", sheet_name="Position")
        for idx, pos in pos_df.iterrows():
            self.Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
    
    def save_tbs(self):
        self.tbsl = Therbligs(self.Pos)
        tbsl_df = pd.read_excel("data.xlsx", sheet_name="Therbligs(L)")   
        self.tbsl.read(tbsl_df)
        
        self.tbsr = Therbligs(self.Pos)
        tbsr_df = pd.read_excel("data.xlsx", sheet_name="Therbligs(R)")   
        self.tbsr.read(tbsr_df)
    
    def save_constraint(self):
        
    
    def gen_OHT(self):
        self.OHT = OHT(self.tbsl, self.tbsr)
        # print(self.OHT)
            
    def run(self):
        self.save_pos()
        self.save_tbs()
        self.gen_OHT()
        print(self.OHT)



#%% Main
myTBHandler = TBHandler()
myTBHandler.run()