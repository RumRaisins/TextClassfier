INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31


class FSM(object):

    NO_STATE = -1
    OPTIONAL_DEFAULT = "default"

    def __init__(self, optional_function,  data=None):
        self.now_state = FSM.NO_STATE
        self.state = set()
        self.optional = set()
        self.optional_function = optional_function
        self.jump = {}
        self.add(data)

    def add(self, data):
        if data is None:
            return
        for src, optionals in data.items():
            # 不存在状态节点
            self._is_exist_state(src)
            for optional in optionals:
                self._is_exist_state(optional["target"])
                self.__is_exist_optional(optional["optional"])
                if src not in self.jump.keys():
                    self.jump[src] = {}
                self.jump[src][optional["optional"]] = optional["target"]

    def _is_exist_state(self, name):
        if name not in self.state:
            self.state.add(name)

    def __is_exist_optional(self, name):
        if name == FSM.OPTIONAL_DEFAULT:
            return
        if name not in self.optional:
            self.optional.add(name)

    def next(self, data):
        if self.now_state == FSM.NO_STATE:
            print("请设置状态")
        if self.optional_function == None:
            print("请设置跳转状态函数 optional_function")

        optional = self.optional_function(data)
        self.now_state = self.jump[self.now_state].get(optional, self.jump[self.now_state][FSM.OPTIONAL_DEFAULT])
        return self.now_state



class Solution1:

    def myAtoi(self, s: str) -> int:
        if len(s) == 0:
            return 0
        def jump(char):
            if char.isspace():
                return "is_space"
            if char == '+' or char == '-':
                return "is_sign"
            if char.isdigit():
                return "is_number"
            return "default"

        fsm = FSM(jump)

        #输入空格跳转到自己
        fsm.add({"start": [{"optional": "is_space", "target": "start"}, {"optional": "is_sign", "target": "signed"},
                          {"optional": "is_number", "target": "number"}, {"optional": "default", "target": "end"}]})

        fsm.add({"signed": [{"optional": "is_number", "target": "number"}, {"optional": "default", "target": "end"}]})

        fsm.add({"number": [{"optional": "is_number", "target": "number"}, {"optional": "default", "target": "end"}]})

        fsm.add({"end": [{"optional": "default", "target": "end"}]})
        fsm.now_state = "start"

        ans, sign = 0, 1
        for c in s:
            fsm.next(c)
            if fsm.now_state == 'number':
                ans = ans * 10 + int(c)
                ans = min(ans, INT_MAX) if sign == 1 else min(ans, -INT_MIN)
            elif fsm.now_state == 'signed':
                sign = 1 if c == '+' else -1
        return ans * sign

class Solution:
    def isNumber(self, s: str) -> bool:
        if len(s) == 0:
            return False

        def jump(char):
            if char.isspace():
                return "is_space"
            elif char == '+' or char == '-':
                return "is_sign"
            elif char == "e" or char =="E":
                return "is_e"
            elif char.isdigit():
                return "is_number"
            elif char == ".":
                return "is_point"
            return "default"

        fsm = FSM(jump)

        fsm.add({"start": [{"optional": "is_space", "target": "start"}, {"optional": "is_sign", "target": "signed"},
                          {"optional": "is_number", "target": "number"}, {"optional": "is_point", "target": "point"},
                           {"optional": "default", "target": "false"}]})
        fsm.add({"number": [{"optional": "is_space", "target": "true"}, {"optional": "is_number", "target": "number"},
                            {"optional": "is_point", "target": "after_point"}, {"optional": "is_e", "target": "E"},
                           {"optional": "default", "target": "false"}]})

        fsm.add({"signed": [{"optional": "is_number", "target": "number"}, {"optional": "is_point", "target": "point"},
                          {"optional": "default", "target": "false"}]})

        fsm.add({"point": [{"optional": "is_number", "target": "after_point"},{"optional": "default", "target": "false"}]})

        fsm.add({"after_point": [{"optional": "is_number", "target": "after_point"},{"optional": "is_e", "target": "E"},
                                 {"optional": "is_space", "target": "true"},{"optional": "default", "target": "false"}]})

        fsm.add({"E": [{"optional": "is_number", "target": "e_number"},{"optional": "is_sign", "target": "e_signed"},
                       {"optional": "default", "target": "false"}]})
        fsm.add({"e_number": [{"optional": "is_number", "target": "e_number"}, {"optional": "is_space", "target": "true"},
                       {"optional": "default", "target": "false"}]})

        fsm.add({"e_signed": [{"optional": "is_number", "target": "e_number"},{"optional": "default", "target": "false"}]})

        fsm.add({"true": [{"optional": "is_space", "target": "true"}]})
        fsm.now_state = "start"
        s += " "
        for c in s:
            fsm.next(c)
            if fsm.now_state == 'false':
                return False
            if fsm.now_state == "true":
                return True
        return False




if __name__ == '__main__':
    s = Solution1()
    print(s.myAtoi("-91283472332"))
    print(s.myAtoi("4193 with words"))
    print(s.myAtoi("words and 987"))
    print(s.myAtoi("+-231"))
    print(s.myAtoi("  -9112"))

    s = Solution()

    true_data = ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
    false_data = ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]

    for data in true_data:
        print(data + " is ", end="")
        print(s.isNumber(data))

    for data in false_data:
        print(data + " is ", end="")
        print(s.isNumber(data))

