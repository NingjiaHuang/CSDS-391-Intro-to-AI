import copy
import random
import itertools
from queue import PriorityQueue

expanded = []

class Node:
    def __init__(self, start_state, gCost, parent=None):
        self.current_state = start_state
        self.gCost = gCost
        self.parent = parent
        self.goal = [
            ['B','1', '2'],
            ['3', '4', '5'],
            ['6', '7', '8']
        ]
        self.hCost = self.calc_heuristic(2)
        self.max = 1000000000000000000000

    def move(self, direction, parameter): 
        row, col = find_position('B',parameter)
        copy2 = copy.deepcopy(parameter)
        if direction == 'u':
            copy2[row][col] = copy2[row-1][col]
            copy2[row-1][col] = 'B'
        elif direction == 'd': 
            copy2[row][col] = copy2[row+1][col]
            copy2[row+1][col] = 'B'
        elif direction == 'l':
            copy2[row][col] = copy2[row][col-1]
            copy2[row][col-1] = 'B'
        elif direction == 'r':
            copy2[row][col] = copy2[row][col+1]
            copy2[row][col+1] = 'B'
        else: 
            raise Exception("Undefined action.")
        return copy2
    
    def print_state(self):
        print(self.current_state)

    def generate_successors(self):
        successors = []
        i,j = find_position('B',self.current_state)
        if i > 0: 
            successors.append(Node(self.move('u',self.current_state),self.gCost+1,self))
        if i < 2: 
            successors.append(Node(self.move('d',self.current_state),self.gCost+1,self))
        if j > 0: 
            successors.append(Node(self.move('l',self.current_state),self.gCost+1,self))
        if j < 2: 
            successors.append(Node(self.move('r',self.current_state),self.gCost+1,self))
        if self.parent == None: 
            return successors
        else: 
            for successor in successors: 
                if state_compare(successor.current_state, successor.parent.parent.current_state):
                    successors.pop(find_node_index(successor, successors))
            return successors

    def is_goal(self):
        return self.current_state == self.goal 

    #will implement heuristic 2 algorithm, which gives the sum of the distances of tiles from their goal positions.
    def calc_heuristic(self, heuristic): 
        hCost = 0
        if(heuristic == 2): 
            if self.is_goal(): 
                return 0
            else: 
                for i in range(len(self.current_state)):
                    for j in range(len(self.current_state[i])):
                        if self.current_state[i][j] != self.goal[i][j]:
                            rep = self.current_state[i][j]
                            i_goal, j_goal = find_position(rep, self.goal)
                            hCost += abs(i_goal - i) + abs(j_goal - j)
            return hCost
        elif (heuristic == 1): 
            for i in range(len(self.current_state)):
                for j in range(len(self.current_state[i])):
                    if (self.current_state[i][j] != self.goal[i][j]):
                        hCost += 1
            return hCost
        else: 
            raise Exception("Undefined heuristic.")

    #implementing a* search, n is the maximum nodes we would like to check. 
    def a_star(self,n):
        expanded.clear()
        print("Searching using A* search...")
        if check_solvable(self,self.goal) == False:
            return
        queue = []
        queue.append(self)
        deal = False
        while(queue):
            config = find_smallest_cost_node(queue)
            if(self.max_node(len(expanded))): 
                expanded.clear()
                return True
            expanded.append(config)
            queue.pop(find_node_index(config,queue))
            if(config.is_goal()):
                print(*config.determine_path(),sep="\n")
                print("Goal state reached, # of moves: ", len(config.determine_path()) - 1)
                expanded.clear()
                return True
            successors = config.generate_successors()
            for successor in successors: 
                if successor in expanded:
                    for node in expanded:
                        if successor == node:
                            if successor.gCost + successor.hCost < node.gCost + node.hCost:
                                node.gCost = successor.gCost
                                node.hCost = successor.hCost
                                node.parent = successor.parent
                                queue.insert(0,node)
                                expanded.pop(find_node_index(node,expanded))
                            deal = True
                if successor in queue: 
                    for element in queue: #condition: when the current node is in queue
                        if successor.current_state == element.current_state: #if unexpanded nodes have the same current state, keep the one with the lower cost
                            if successor.gCost + successor.hCost < element.gCost + element.hCost: 
                                element.gCost = successor.gCost
                                element.hCost = successor.hCost
                                element.parent = successor.parent
                            deal = True
                if(deal==False):
                    queue.append(successor)

    def __lt__(self, other):
        self.gCost + self.hCost < other.gCost + other.hCost

    def beam_search(self, k, n):
        print("Searching using local beam search...")
        if check_solvable(self,self.goal) == False:
            return
        found = False
        explored_map = {} # store the explored nodes
        q = PriorityQueue() # unexplored nodes
        q.put((self.gCost + self.hCost, self))
        num_counter = 0
        while (q.empty() == False and (not found)): 
            while q.empty() == False:
                if(self.max_node(num_counter)):
                    return
                tupl = q.get()
                current_node = tupl[1]
                if tuple(map(tuple, current_node.current_state)) not in explored_map.keys(): 
                    explored_map[tuple(map(tuple,current_node.current_state))] = current_node
                else: 
                    continue
                if(current_node.is_goal()):
                    found = True
                    print(*current_node.determine_path(),sep="\n")
                    print("Goal state reached, # of moves: ", len(current_node.determine_path()))
                    return True
                else: 
                    successors = current_node.generate_successors()
                    for successor in successors: 
                        tup = tuple(map(tuple,successor.current_state))
                        if explored_map.get(tup) != None: 
                            successors.pop(find_node_index(successor,successors))
                        elif successor.current_state == successor.goal: 
                            found = True
                            print(*current_node.determine_path(),sep="\n")
                            print(successor.current_state)
                            print("Goal state reached, # of moves: ", len(current_node.determine_path()))
                            return True
                        else:
                            pass
                    num_counter += 1
                    for i in successors: 
                        q.put((i.gCost + i.hCost, i))
                    
    #retrieve the path when goal state is founded. 
    def determine_path(self):
        path = [self.current_state]
        pointer = self
        while(pointer.parent != None):
            pointer = pointer.parent
            path.insert(0,pointer.current_state)
        return path

    def set_state(self, string1, string2, string3): 
        init_state = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        for i in range(3): 
            init_state[0][i] = string1[i]
            init_state[1][i] = string2[i]
            init_state[2][i] = string3[i]           
        self.current_state = init_state

    #specify the maximum number of nodes that can be expanded during the search
    def max_node(self, current_num):
        if current_num > self.max:
            print("Visited nodes exceed the maximum limit. Max limit: ", self.max)
        return current_num > self.max


#count the number of inversions in order to tell whether a state is solvable or not
def check_solvable(node,goal_state):
    num_inversion = 0
    origin_rep = list(itertools.chain(*node.current_state))
    origin_rep.remove('B')
    origin_rep = [int(i) for i in origin_rep] 
    goal_rep = list(itertools.chain(*goal_state))
    goal_rep.remove('B')
    goal_rep = [int(i) for i in goal_rep] 
    for i in range(len(origin_rep)):
        for j in range(len(goal_rep)):
            if(origin_rep[i] == goal_rep[j]):
                for i_pointer in range(i+1,len(origin_rep)):
                    if i < len(origin_rep)-1:
                        if(origin_rep[i_pointer] in goal_rep[:j]):
                            num_inversion += 1
    parity = num_inversion % 2
    if(parity == 0):
        return True
    else: 
        print("The puzzle given is unsolvable.")
        return False

def find_node_index(node, list): 
    for i in range(len(list)):
        if node == list[i]:
            return i
    return -1

#find the node with the smallest estimated cost: f(n) = g(n) + h(n)
def find_smallest_cost_node(node_list): 
    fCost = float("inf")
    node = node_list[0]
    for current_node in node_list: 
        if current_node.gCost + current_node.hCost < fCost: 
            fCost = current_node.gCost + current_node.hCost
            node = current_node
    return node

#find the index of the tile in the current state
def find_position(toFind, state): 
    for outer in range(len(state)): # outer for row, inner for column
        for inner in range(len(state[outer])):
            if state[outer][inner] == toFind:
                return (outer, inner) 
    return (-1, -1)

#random.seed(0)
#make random moves from the goal state to make sure a solution exist.
def randomize_state(n,node):
    node.current_state = node.goal
    movement_list = ['u','d','l','r']
    movement_history = []
    for k in range(n):
        integer = random.randint(0,3)
        i,j = find_position('B',node.current_state)
        while (i == 0 and integer == 0) or (i == 2 and integer == 1) or (j == 0 and integer == 2) or (j == 2 and integer == 3): 
            integer = random.randint(0,3)
        node.current_state = node.move(movement_list[integer],node.current_state)
        movement_history.append(movement_list[integer])
    if(node.current_state == node.goal):
        return randomize_state(n,node)
    else: 
        print("Generated state: ", node.current_state)
        print("The movements from goal been taken is: ", movement_history)
        return node.current_state

def sort_array(node_list):
    for i in range(len(node_list) - 1):
        if(node_list[i].hCost > node_list[i+1].hCost):
            temp = node_list[i]
            node_list[i] = node_list[i+1]
            node_list[i+1] = temp
    return node_list

def state_compare(state,goal):
	for i in range(len(state)):
		for j in range(len(state)):
			if state[i][j] != goal[i][j]:
				return False
	return True

def main():
    node = Node([['B','1','2'],['3','4','5'],['6','7','8']],0)
    file = open('text.txt','r')
    lines = file.readlines()
    for line in lines: 
        line = line.strip()
        string = line.split(' ')
        if string[0] == "setState": 
            node.set_state(string[1],string[2],string[3])
            print(node.current_state)
        elif string[0] == "printState": 
            node.print_state()
        elif string[0] == "move":
            direction = ''
            if string[1] == "up": 
                direction = 'u'
            elif string[1] == "down":
                direction = 'd'
            elif string[1] == "left":
                direction = 'l'
            elif string[1] == "right":
                direction = 'r'
            else: 
                print("Wrong direction.")
            node.current_state = node.move(direction,node.current_state)
        elif string[0] == "randomizeState":
            randomize_state(int(string[1]),node)
        elif string[0] == "solve":
            method = string[1]
            if string[2] == "h1": 
                heuristic = 1
            if string[2] == "h1":
                heuristic = 1
            elif string[2] == "h2":
                heuristic = 2
            else: 
                heuristic = 0
            if method == "A-star": 
                node.hCost = node.calc_heuristic(heuristic)
                node.a_star(100000000)
            elif method == "beam":
                node.beam_search(int(string[2]), 10000000)
        elif string[0] == "maxNode": 
            node.max = int(string[1])
        else:
            print("Not proper commands. Please follow the following input structure: ")
            print("setState state")
            print("printState")
            print("move direction")
            print("randomizeState n")
            print("solve A-star heuristic")
            print("solve beam k")
            print("maxNode n")

def test_3d(node): 
    num_count = 0
    exist_map = {}
    i = 0
    for i in range(51):
        test_case = randomize_state(30,node)
        if tuple(map(tuple,test_case)) not in exist_map: 
            exist_map[tuple(map(tuple,test_case))] = True
            current_node = Node(test_case,0)
            if current_node.a_star(1000000000):
                num_count += 1
        else: 
            i -= 1
    return num_count
        
if __name__ == "__main__": 
    main()
