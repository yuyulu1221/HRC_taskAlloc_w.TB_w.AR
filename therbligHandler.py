
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
        
    def read(self, df: pd.DataFrame):
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
    def __init__(self, ls:list):
        self.tb_list = ls
        self.in_edge = []
        self.out_edge = []
        self.bind_edge = []
            
    def __repr__(self):
        return "[" + ", ".join(map(str, self.tb_list)) + "]"
    
    # Handle overlap time
    def set_relation(self, overlap=-1):
        self.overlap = overlap
    
#%% TBHandler
class TBHandler(object):
    def __init__(self):
        self.Pos = {}
        self.tbsl:Therbligs
        self.tbsr:Therbligs
        self.OHT_list = []

    def save_pos(self):
        pos_df = pd.read_excel("data1.xlsx", sheet_name="Position")
        for idx, pos in pos_df.iterrows():
            self.Pos[pos["Name"]] = np.array([float(pos["x_coord"]), float(pos["y_coord"]),float(pos["z_coord"])])
    
    # Save tbs by list
    def save_tbs(self):
        self.tbsl = Therbligs(self.Pos) # save pos
        tbsl_df = pd.read_excel("data1.xlsx", sheet_name="Therbligs(L)")   
        self.tbsl.read(tbsl_df)
        
        self.tbsr = Therbligs(self.Pos)
        tbsr_df = pd.read_excel("data1.xlsx", sheet_name="Therbligs(R)")   
        self.tbsr.read(tbsr_df)
        
    # Convert tbs to oht    
    def read_tbs(self):
        for tbs in (self.tbsl, self.tbsr):
            if not isinstance(tbs, Therbligs):
                continue
            else:
                tmp = []
                for tb in tbs.list:
                    if len(tmp) != 0:
                        if tmp[-1] == "A" or tmp[-1].type == "DA" and tb.type != "RL":
                            raise ValueError("Invalid therblig list")  
                    tmp.append(tb)
                    if tb.type == "RL":
                        # self.list.append(tmp.copy())
                        self.OHT_list.append(OHT(tmp.copy()))
                        tmp.clear()
            
    def run(self):
        self.save_pos()
        self.save_tbs()
        self.read_tbs()
        # self.create_OHT()
        print(self.OHT_list)



#%% Main
myTBHandler = TBHandler()
myTBHandler.run()