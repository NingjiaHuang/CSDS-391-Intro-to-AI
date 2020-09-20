import copy

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
        self.move = " "
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
    
    def printState(self):
        print(self.current_state)

    def generateSuccessors(self):
        successors = []
        i,j = find_position('B',self.current_state)
        if i > 0: 
            successors.append(Node(self.move('u'),self.gCost+1,self))
            self.move = 'u'
        if i < 2: 
            successors.append(Node(self.move('d'),self.gCost+1,self))
            self.move = 'd'
        if j > 0: 
            successors.append(Node(self.move('l'),self.gCost+1,self))
            self.move = 'l'
        if j < 2: 
            successors.append(Node(self.move('r'),self.gCost+1,self))
            self.move = 'r'
        if self.parent == None: 
            return successors
        else: 
            for successor in successors: 
                for element in expanded:
                    if successor.current_state == element.current_state:
                        successors.pop(find_node_index(successor, successors))                        
        return successors

    def isGoal(self):
        if self.current_state == self.goal: 
            return True
        else: 
            return False

    # will implement heuristic 2 algorithm, which gives the sum of the distances of tiles from their goal positions.
    def calc_heuristic(self): 
        hCost = 0
        if self.isGoal(): 
            return 0
        else: 
            for i in range(len(self.current_state)):
                for j in range(len(self.current_state[i])):
                    if self.current_state[i][j] != self.goal[i][j]:
                        rep = self.current_state[i][j]
                        i_goal, j_goal = find_position(rep, self.goal)
                        hCost += abs(i_goal - i) + abs(j_goal - j)
        return hCost

    # implementing a* search
    def a_star(self):
        queue = []
        queue.append(self)
        deal = False
        while(len(queue) != 0):
            config = find_smallest_cost_node(queue)
            expanded.append(config)
            queue.pop(find_node_index(config,queue))
            if(config.isGoal()):
                print(*config.determine_path(),sep="\n")
                print("Goal state reached, # of move: ", len(config.determine_path()))
                return
            else: 
                successors = config.generateSuccessors()
                for successor in successors: 
                    #print("successor: ", successor.current_state)
                    for node in expanded:
                        #print("node: ", node.current_state)
                        if successor.current_state == node.current_state:
                            #print("yes")
                            if successor.gCost + successor.hCost < node.gCost + node.hCost:
                                node.gCost = successor.gCost
                                node.hCost = successor.hCost
                                node.parent = successor.parent
                                queue.insert(0,node)
                                expanded.pop(find_node_index(node,expanded))
                            deal = True

                    for element in queue: # condition: when the current node is in queue
                        if successor.current_state == element.current_state: # if unexpanded nodes have the same current state, keep the one with the lower cost
                            if successor.gCost + successor.hCost < element.gCost + element.hCost: 
                                element.gCost = successor.gCost
                                element.hCost = successor.hCost
                                element.parent = successor.parent
                            deal = True

                    if(not deal):
                        queue.append(successor)
    # retrieve the path when goal state is founded. 
    def determine_path(self):
        path = [self.current_state]
        pointer = self
        while(pointer.parent != None):
            pointer = pointer.parent
            path.insert(0,pointer.current_state)
        return path

def setState(puzzle): 
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

# find the node with the smallest estimated cost: f(n) = g(n) + h(n)
def find_smallest_cost_node(node_list): 
    fCost = float("inf")
    node = node_list[0]
    for current_node in node_list: 
        if current_node.gCost + current_node.hCost < fCost: 
            fCost = current_node.gCost + current_node.hCost
            node = current_node
    return node

# find the index of the tile in the current state
def find_position(toFind, state): 
    for outer in range(len(state)): # outer for row, inner for column
        for inner in range(len(state[outer])):
            if state[outer][inner] == toFind:
                return (outer, inner) 
    return (-1, -1)

def main():
    config = setState('test.txt')
    node = Node(config, 0)
    node.a_star()
    #successors = node.generateSuccessors()
    #test_node = successors[0]
    #Xprint(successors.pop(find_node_index(test_node,successors)).current_state)

if __name__ == "__main__": 
    main()
