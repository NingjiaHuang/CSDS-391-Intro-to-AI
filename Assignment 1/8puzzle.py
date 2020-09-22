import copy
import random
expanded = []

class Node:
    def __init__(self, start_state, gCost,parent=None):
        self.current_state = start_state
        self.gCost = gCost
        self.parent = parent
        self.goal = [
            ['B','1', '2'],
            ['3', '4', '5'],
            ['6', '7', '8']
        ]
        self.hCost = self.calc_heuristic()
    
    def move(self, direction): 
        row, col = find_position('B',self.current_state)
        copy2 = copy.deepcopy(self.current_state)
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
            successors.append(Node(self.move('u'),self.gCost+1,self))
        if i < 2: 
            successors.append(Node(self.move('d'),self.gCost+1,self))
        if j > 0: 
            successors.append(Node(self.move('l'),self.gCost+1,self))
        if j < 2: 
            successors.append(Node(self.move('r'),self.gCost+1,self))
        if self.parent == None: 
            return successors
        else: 
            for successor in successors: 
                for element in expanded:
                    if successor.current_state == element.current_state:
                        successors.pop(find_node_index(successor, successors))                        
        return successors

    def is_goal(self):
        if self.current_state == self.goal: 
            return True
        else: 
            return False

    #will implement heuristic 2 algorithm, which gives the sum of the distances of tiles from their goal positions.
    def calc_heuristic(self): 
        hCost = 0
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

    #implementing a* search
    def a_star(self,n):
        queue = []
        queue.append(self)
        deal = False
        while(len(queue) != 0):
            config = find_smallest_cost_node(queue)
            if(max_node(n, len(expanded))): 
                return
            expanded.append(config)
            queue.pop(find_node_index(config,queue))
            if(config.isGoal()):
                print(*config.determine_path(),sep="\n")
                print("Goal state reached, # of moves ", len(config.determine_path()))
                return
            else: 
                successors = config.generate_successors()
                for successor in successors: 
                    for node in expanded:
                        if successor.current_state == node.current_state:
                            if successor.gCost + successor.hCost < node.gCost + node.hCost:
                                node.gCost = successor.gCost
                                node.hCost = successor.hCost
                                node.parent = successor.parent
                                queue.insert(0,node)
                                expanded.pop(find_node_index(node,expanded))
                            deal = True

                    for element in queue: #condition: when the current node is in queue
                        if successor.current_state == element.current_state: #if unexpanded nodes have the same current state, keep the one with the lower cost
                            if successor.gCost + successor.hCost < element.gCost + element.hCost: 
                                element.gCost = successor.gCost
                                element.hCost = successor.hCost
                                element.parent = successor.parent
                            deal = True

                    if(not deal):
                        queue.append(successor)

    def beam_search(self, k, n):
        found = False
        queue = [] 
        children_k = []
        queue.append(self)
        num_counter = 1
        while(len(queue) != 0 and found == False):
            #pop all nodes in queue and store their successors in a list
            while(len(queue) != 0):        
                if(max_node(n,num_counter)):
                    return
                current_node = queue.pop()
                if(current_node.is_goal()):
                    found = True
                    print(*current_node.determine_path(),sep="\n")
                    print("Goal state reached, # of moves: ", len(current_node.determine_path()))
                    return 
                else: 
                    successors = current_node.generate_successors()
                    num_counter += 1
                    children_k.extend(successors)
            children_k = sort_array(children_k)
            children_k = children_k[:k]
            for j in range(len(children_k)):
                if(children_k[j].current_state == children_k[j].goal):
                    found = True
                    print(*children_k[j].determine_path(),sep="\n")
                    print("Goal state reached, # of moves: ", len(current_node.determine_path()))
                    return 
            if(found == False):
                queue.extend(children_k)
                children_k.clear()

    #retrieve the path when goal state is founded. 
    def determine_path(self):
        path = [self.current_state]
        pointer = self
        while(pointer.parent != None):
            pointer = pointer.parent
            path.insert(0,pointer.current_state)
        return path

def set_state(puzzle): 
    init_state = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    file = open(puzzle,'r')
    row = 0
    col = 0
    line = file.readline()
    for character in line: 
        if character == ' ': 
            continue
        else:
            if col < 3:
                init_state[row][col] = character
                col += 1
            else: 
                col = 0
                row += 1
                init_state[row][col] = character
                col += 1
    return init_state

def find_node_index(node, list): 
    for i in range(len(list)):
        if node == list[i]:
            return i
    return -1

#specify the maximum number of nodes that can be expanded during the search
def max_node(expected_num,current_num):
    if current_num > expected_num:
        print("Visited nodes exceed the maximum limit. Max limit:", expected_num)
    return current_num > expected_num

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

#make random moves from the goal state to make sure a solution exist.
def randomize_state(n,node):
    movement_list = ['u','d','l','r']
    for k in range(n):
        integer = random.randint(0,3)
        i,j = find_position('B',node.current_state)
        while (i == 0 and integer == 0) or (i == 2 and integer == 1) or (j == 0 and integer == 2) or (j == 2 and integer == 3): 
            integer = random.randint(0,3)
        node.current_state = node.move(movement_list[integer])
        print(node.current_state, ' ', movement_list[integer])
    return node

def sort_array(node_list):
    for i in range(len(node_list) - 1):
        if(node_list[i].hCost > node_list[i+1].hCost):
            temp = node_list[i]
            node_list[i] = node_list[i+1]
            node_list[i+1] = temp
    return node_list

def main():
    config = set_state('test.txt')
    node = Node(config, 0)
    #node.beam_search(30,500000000)
    #node.a_star(20)

if __name__ == "__main__": 
    main()
