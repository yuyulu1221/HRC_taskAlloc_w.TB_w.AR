#%% 
import numpy as np
import pandas as pd
import copy

#%% Test
tb_abbr = {
    "R": "Reach",
    "M": "Move",
    "G": "Grasp",
    "RL": "Release Load",
    "A": "Assemble",
    "DA": "Disassemble",
    "P": "Position",
    "H": "Hold"
}

AGENT = ["LH", "RH", "BOT"]


#%% 動素
class Therblig(object):
    def __init__(self, Name:str=None, From:str=None, To:str=None, Type:str=None, Obj1:str=None, Obj2:str=None):
        if tb_abbr.get(Name) == None:
            raise ValueError("This type of therblig is not exist")
        self.name = Name
        self.start: float
        self.end: float
        self.From = From
        self.To = To
        self.Type = Type
        self.Obj1 = Obj1
        self.Obj2 = Obj2
        self.time: float
        # self.next = None
        # self.tb_process_time = pd.read_excel("data_test.xlsx", sheet_name="Therblig Process Time")   
        
    def __repr__(self):
        return f"#{str(self.name)}"
    
    def get_tb_time(self, agent, agent_POS, POS, MTM):
        # print(f"MTM.loc[{self.type}, {AGENT[agent]}] * {np.lonalg.norm}")
        if self.name in ["R", "M"]:
            dist = (np.linalg.norm(POS[self.To] - POS[AGENT[agent]]) + 2) // 2 * 2
            # set agent to new position
            agent_POS[AGENT[agent]] = POS[self.To]
            dist = dist if dist < 28 else 28
            return MTM.at[self.name + str(int(dist)) + self.Type, AGENT[agent]]
            # return MTM.at[self.type, AGENT[agent]] * np.linalg.norm(POS[self.To] - POS[self.From])
        else:
            return MTM.at[self.name, AGENT[agent]]

#%% 動素序列
class RawTherbligList(object):
    def __init__(self, Pos):
        self.list = []
        self.Pos = Pos
    def __repr__(self):
        return ", ".join(map(str, self.list))
        
    def read(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            therblig = Therblig(
                Name = row["Name"], 
                From = row["From"], 
                To = row["To"], 
                Type = row["Type"],
                Obj1 = row["Obj1"], 
                Obj2 = row["Obj2"]
            )
            self.list.append(copy.copy(therblig))


#%% 單手任務
class OHT:
    def __init__(self, ls:list):
        self.id = -1
        self.tb_list = ls
        self.next = []
        self.prev = []
        self.bind = -1
        # self.is_scheduled = False
            
    def __repr__(self):
        return "(" + ", ".join(map(str, self.tb_list)) + ")"
        # return f"OHT{self.id}"
    
    def set_id(self, id):
        self.id = id
    
    def add_next(self, n):
        self.next.append(n)
    
    def add_prev(self, p):
        self.prev.append(p)
    
    def get_oht_time(self, agent, agent_pos, pos, mtm):
        # print("????")
        oht_t = 0
        for tb in self.tb_list:
            oht_t += tb.get_tb_time(agent, agent_pos, pos, mtm)
        return oht_t
        
#%% TBHandler
class TBHandler(object):
    def __init__(self):
        self.Pos = {}
        self.tbsl:RawTherbligList
        self.tbsr:RawTherbligList
        self.OHT_list = []
    
    ## Save tbs by list
    def save_tbs(self):
        self.tbsl = RawTherbligList(self.Pos) # save pos
        tbsl_df = pd.read_excel("data2.xlsx", sheet_name="Therbligs(L)")   
        self.tbsl.read(tbsl_df)
        
        self.tbsr = RawTherbligList(self.Pos)
        tbsr_df = pd.read_excel("data2.xlsx", sheet_name="Therbligs(R)")   
        self.tbsr.read(tbsr_df)
        
    ## Convert tbs to oht    
    def read_tbs(self):
        for tbs in (self.tbsl, self.tbsr):
            if not isinstance(tbs, RawTherbligList):
                continue
            else:
                tmp = []
                for tb in tbs.list:
                    if len(tmp) != 0:
                        if (tmp[-1].name == "A" or tmp[-1].name == "DA") and tb.name != "RL":
                            raise ValueError("Invalid therblig list")  
                    tmp.append(tb)
                    if tb.name == "RL":
                        # self.list.append(tmp.copy())
                        self.OHT_list.append(OHT(tmp.copy()))
                        tmp.clear()
        # Use dummy node to represent "END"
        self.OHT_list.append(OHT([]))
                        
    def get_oht_time(self, oht_id, Pos):
        self.OHT_list[oht_id].get_oht_time(Pos)
           
    def set_oht_id(self):
        for id, oht in enumerate(self.OHT_list):
            oht.set_id(id)
        for oht in self.OHT_list:
            print("id: ", oht.id)
     
    def run(self):
        self.save_tbs()
        self.read_tbs()
        self.set_oht_id()
        print(self.OHT_list)



#%% Main
# myTBHandler = TBHandler()
# myTBHandler.run()