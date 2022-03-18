import tkinter

import networkx as nx
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib.figure import Figure
from JSAnimation.IPython_display import display_animation, anim_to_html

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import *
from collections import deque
from heapq import heappop, heappush
import matplotlib.animation as animation
from math import inf, sqrt
from itertools import combinations
from random import random
import time

g = nx.DiGraph()


class Graph:
    def __init__(self, directed=True):
        self.edges = {}
        self.huristics = {}
        self.nodes = []
        self.directed = directed

    def addnode(self, nodename):
        if nodename not in self.nodes:
            self.nodes.append(nodename)

    def cleargraph(self, directed=True):
        self.edges = {}
        self.huristics = {}
        self.nodes = []
        self.directed = directed

    """
    def addingthenode(self, node1, node2, x1, x2, y1, y2, reversed):

        nodeone = [x1, y1]
        nodetwo = [x2, y2]
        self.coordinates[node1] = nodeone
        self.coordinates[node2] = nodetwo
        c = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.add_edge(node1, node2, int(c), reversed)
    """

    def add_edge(self, node1, node2, cost=1, __reversed=False):
        cost = int(cost)
        self.addnode(node1)
        self.addnode(node2)
        try:
            neighbors = self.edges[node1]
        except KeyError:
            neighbors = {}
        neighbors[node2] = cost
        self.edges[node1] = neighbors

        if not self.directed and not __reversed: self.add_edge(node2, node1, cost, True)

    """
    def create_huristics(self, goal):
        Gx = self.coordinates[goal][0]
        Gy = self.coordinates[goal][1]

        for node in self.coordinates.keys():
            X1 = self.coordinates[node][0]
            Y1 = self.coordinates[node][1]
            Z = ((Y1 - Gy) * (Y1 - Gy)) + ((Gx - X1) * (Gx - X1))
            self.huristics[node] = int(sqrt(Z))
    """

    def createhuristic(self, node1, heur1):
        self.huristics[node1] = heur1

    def set_huristics(self, huristics={}):
        self.huristics = huristics

    def neighbors(self, node):
        try:
            return self.edges[node]
        except KeyError:
            return []

    def cost(self, node1, node2):
        try:
            return self.edges[node1][node2]
        except:
            return inf

    def ass(self, start, goal):  # a star search

        found = False  # condition if we found the goal or not which will be one of the conditions that
        # will stop iterating over the graph
        opened = [(self.huristics[start], start)]  # a list of the frontier or the nodes that are alligable to be
        # explored and by heap sort we get the node that has highest priority contains tuple of a pair which is the
        # heuristic value to be added and the rest of the path(cant be changed to ensure security
        closed = set(
            start)  # setting the visited nodes to eliminate going into an infinit loop and making into a set as node
        # gets explored once so we only need one entrance and lateron we justcheck that node
        came_from = {
            start: None}  # dictionary to keep track of parent to determine the final path where the key is the node and the value is
        cost_so_far = {
            start: 0}  # dictinory tokeep track of the costwhere the keyis the node and the value is the cost of the path
        traversal.insert(tkinter.END, '{:11s} | {} \n'.format('Expand Node', 'Fringe'))
        # in the first 11 spaces print "expanded node" then "|"then in the remaining space print "fringe"
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', str(opened[0])))
        # in the first 11 spaces print"-" then "|" then turn the first position in the opened

        n = 1  # N. . .
        sawa = [x[n] for x in opened]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        counter = 0
        while not found and len(
                opened):  # we will keep exploring the graph till we find the goal or the graph is finished thus the length of the opened list is zero
            counter = counter + 1
            _, node = heappop(  # elsharta tanesh awel return
                opened)  # heappop returns the index and value of node with least cost but we only need the node so we ignore index thus the -
            traversal.insert(tkinter.END, '{:11s} |'.format(node))
            # prints the node being explored in the first 11 spaces and after puts |
            if node in goal: found = True; break  # if node being explored is the goal then no need for continuing the iteration thus break to get out of loop and
            # changing found to true so we dont enter the loop again
            for neighbor in self.neighbors(
                    node):  # looping on the nodes hat are children to the node toadd tothe frontier
                new_cost = cost_so_far[node] + self.cost(node,
                                                         neighbor)  # calculating the cost of each neighbor to determine priority of nodes in frontier
                if neighbor not in closed :
                    closed.add(neighbor);  # marking node as explored
                    # the value determined will onlybe used if the node is not already explored
                    # and if it it is value ishould be less than the value already assigned to the neighbour to be used
                    came_from[neighbor] = node;  # and setting the  parent of the neighbour of the node to node
                    # (again this will only happen if the parent(node)has a less cost than the prior parent)
                    cost_so_far[
                        neighbor] = new_cost  # as the parent of the node was changed the cost of the path changes thus we update it
                    heappush(opened, (new_cost + self.huristics[neighbor],
                                      neighbor))
                elif cost_so_far[
                    neighbor] > new_cost:# adding the neighbour to the list to be explored with its weight as
                    came_from[neighbor] = node;  # and setting the  parent of the neighbour of the node to node
                    # (again this will only happen if the parent(node)has a less cost than the prior parent)
                    cost_so_far[neighbor] = new_cost                    # the cost of the path + cost of heuristic

            traversal.insert(tkinter.END, ', '.join([str(n) for n in opened]))
            traversal.insert(tkinter.END, '\n')
            # printing fringe
            sawa = [x[n] for x in opened]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[node], node,iteration  # returning thepath and the cost if the function finds the goal
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}'.format(start, goal))
            return None, inf, None,iteration  # goal is not in tree

    def greedy_search(self, start, goal):
        found, fringe, visited, came_from, cost_so_far = False, [(self.huristics[start], start)], set([start]), {
            start: None}, {start: 0}
        n = 1  # N. . .
        sawa = [x[n] for x in fringe]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        traversal.insert(tkinter.END, '{:11s} | {} \n'.format('Expand Node', 'Fringe'))
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', str(fringe[0])))
        counter = 0
        while not found and len(fringe):
            counter = counter + 1
            _, current = heappop(fringe)
            traversal.insert(tkinter.END, '{:11s} |'.format(current))
            if current in goal: found = True; break
            for node in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, node)
                if node not in visited :
                    visited.add(node);
                    came_from[node] = current;
                    cost_so_far[node] = new_cost
                    heappush(fringe, (self.huristics[node], node))
                elif cost_so_far[node] > new_cost:
                    came_from[node] = current;
                    cost_so_far[node] = new_cost
            traversal.insert(tkinter.END, ', '.join([str(n) for n in fringe]))
            sawa = [x[n] for x in fringe]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
            traversal.insert(tkinter.END, '\n')

        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[current], current,iteration
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}'.format(start, goal))
            return None, inf, None,iteration

    def depth_first_search(self, start, goal):
        found, fringe, visited, came_from, cost_so_far = False, deque([start]), [start], {start: None}, {start: 0}
        traversal.insert(tkinter.END, '{:11s} | {} \n'.format('Expand Node', 'Fringe'))
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', start))
        n = 0  # N. . .
        sawa = [x[n] for x in fringe]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        counter=0
        while not found and len(fringe):
            counter = counter + 1
            current = fringe.pop()
            traversal.insert(tkinter.END, '{:11s} |'.format(current))
            if current in goal: found = True; break
            for node in self.neighbors(current):

                new_cost = cost_so_far[current] + self.cost(current, node)
                if node not in visited:
                    visited.append(node);
                    fringe.append(node);
                    came_from[node] = current
                    cost_so_far[node] = new_cost
            traversal.insert(tkinter.END, ','.join(fringe))
            traversal.insert(tkinter.END, '\n')
            sawa = [x[n] for x in fringe]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
        # showgraph(frontier=[])
        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[current], current,iteration
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}'.format(start, goal))
            animate(iteration)

    def breadth_first_search(self, start, goal):
        found, fringe, visited, came_from, cost_so_far = False, deque([start]), set([start]), {start: None}, {start: 0}
        traversal.insert(tkinter.END, '{:11s} | {} \n'.format('Expand Node', 'Fringe'))
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', start))
        n = 0  # N. . .
        sawa = [x[n] for x in fringe]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        counter=0
        while not found and len(fringe):
            counter=counter+1
            current = fringe.pop()
            traversal.insert(tkinter.END, '{:11s} |'.format(current))
            if current in goal: found = True; break
            for node in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, node)
                if node not in visited:
                    visited.add(node);
                    fringe.appendleft(node);
                    came_from[node] = current
                    cost_so_far[node] = new_cost
            traversal.insert(tkinter.END, ','.join(fringe))
            traversal.insert(tkinter.END, '\n')
            sawa = [x[n] for x in fringe]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[current], current,iteration
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}'.format(start, goal))
            animate(iteration)

    def depth_limited_search(self, start, goal, limit=-1):
        traversal.insert(tkinter.END, 'Depth limit = {}\n'.format(limit))
        found, fringe, visited, came_from, cost_so_far = False, deque([(0, start)]), set([start]), {start: None}, {
            start: 0}
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('Expand Node', 'Fringe'))
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', start))
        n = 1  # N. . .
        sawa = [x[n] for x in fringe]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        counter=0
        while not found and len(fringe):
            counter=counter+1
            depth, current = fringe.pop()
            traversal.insert(tkinter.END, '{:11s} |'.format(current))
            if current in goal: found = True; break
            if limit == -1 or depth < limit:
                for node in self.neighbors(current):
                    new_cost = cost_so_far[current] + self.cost(current, node)
                    if node not in visited:
                        visited.add(node);
                        fringe.append((depth + 1, node))
                        came_from[node] = current
                        cost_so_far[node] = new_cost
            traversal.insert(tkinter.END, ', '.join([n for _, n in fringe]))
            traversal.insert(tkinter.END, '\n')
            sawa = [x[n] for x in fringe]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[current], current,iteration
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}\n'.format(start, goal))
            return None, visited, None,iteration

    def iterative_deepening_dfs(self, start, goal):
        prev_iter_visited, depth = [], 0

        while True:
            traced_path, visited, end,iteration = self.depth_limited_search(start, goal, depth)
            if traced_path or len(visited) == len(prev_iter_visited):
                return traced_path, visited, end,iteration
            else:
                prev_iter_visited = visited;
                depth += 1

    def uniform_cost_search(self, start, goal):
        found, fringe, visited, came_from, cost_so_far = False, [(0, start)], set([start]), {start: None}, {start: 0}
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('Expand Node', 'Fringe'))
        traversal.insert(tkinter.END, '--------------------\n')
        traversal.insert(tkinter.END, '{:11s} | {}\n'.format('-', str((0, start))))
        n = 1  # N. . .
        sawa = [x[n] for x in fringe]
        colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
        iteration = {}
        iteration[0] = colors
        counter=0
        while not found and len(fringe):
            counter=counter+1
            _, current = heappop(fringe)
            traversal.insert(tkinter.END, '{:11s} |'.format(current))
            if current in goal: found = True; break
            for node in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, node)
                if node not in visited :
                    visited.add(node);
                    came_from[node] = current;
                    cost_so_far[node] = new_cost
                    heappush(fringe, (new_cost, node))
                elif  cost_so_far[node] > new_cost:
                    came_from[node] = current;
                    cost_so_far[node] = new_cost
            traversal.insert(tkinter.END, ', '.join([str(n) for n in fringe]))
            traversal.insert(tkinter.END, '\n')
            sawa = [x[n] for x in fringe]
            colors = [1.0 if x in sawa else 0.0 for x in g.nodes]
            iteration[counter] = colors
        if found:
            traversal.insert(tkinter.END, '\n')
            return came_from, cost_so_far[current], current,iteration
        else:
            traversal.insert(tkinter.END, 'No path from {} to {}'.format(start, goal))
            return None, inf, None,iteration

    @staticmethod
    def print_path(came_from, goal):
        parent = came_from[goal]
        if parent:
            Graph.print_path(came_from, parent)
        else:
            traversal.insert(tkinter.END, goal, "")
            return
        traversal.insert(tkinter.END, ' =>' + goal)

    def __str__(self):
        return str(self.edges)


graph = Graph(directed=True)


def astar():
    traversal.insert(tkinter.END, "\nastar search\n")
    goal = x.get().split(" ")
    traced_path, cost, end,iteration = graph.ass(str(start.get()), goal)

    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)


def greedy():
    traversal.insert(tkinter.END, "\ngreedy search\n")
    goal = x.get().split(" ")

    traced_path, cost, end, iteration = graph.greedy_search(str(start.get()), goal)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)


def breadth():
    traversal.insert(tkinter.END, "\nbreadth search\n")
    goal = x.get().split(" ")
    traced_path, cost, end,iteration = graph.breadth_first_search(str(start.get()), goal)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)


def depthlimited():
    traversal.insert(tkinter.END, "\ndepth limited search\n")
    goal = x.get().split(" ")
    traced_path, cost, end,iteration = graph.depth_limited_search(str(start.get()), goal, 3)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)


def depth():
    traversal.insert(tkinter.END, "\ndepth first search\n")
    goal = x.get().split(" ")
    traced_path, cost, end ,iteration= graph.depth_first_search(str(start.get()), goal)
    # showgraph(frontier=traced_path)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)


def uniform():
    traversal.insert(tkinter.END, "\nucs search\n")
    goal = x.get().split(" ")
    traced_path, cost, end,iteration= graph.uniform_cost_search(str(start.get()), goal)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)

def iterative():
    traversal.insert(tkinter.END, "\niterative deepening search\n")
    goal = x.get().split(" ")
    traced_path, cost, end,iteration = graph.iterative_deepening_dfs(str(start.get()), goal)
    if (traced_path):
        Graph.print_path(traced_path, end);
        traversal.insert(tkinter.END, '{:11s} {}'.format('\nCost:', cost))
    animate(iteration)

def enteredge():
    try:
        Convert_To_Intcost = int(e3.get())
        Convert_To_Intheuristicnode1 = int(e4.get())
        Convert_To_Intheuristicnode2 = int(e5.get())

        errormess.place_forget()
    except ValueError:
        errormess.place(x=350, y=120)
        return
    # c = sqrt((Convert_To_Intx1 - Convert_To_Intx2) ** 2 + (Convert_To_Inty1 - Convert_To_Inty2) ** 2)
    startnode = str(e1.get())
    endingnode = str(e2.get())
    graph.createhuristic(startnode, Convert_To_Intheuristicnode1)
    graph.createhuristic(endingnode, Convert_To_Intheuristicnode2)
    g.add_node(startnode)
    g.add_node(endingnode)

    g.add_edge(startnode, endingnode, weight=Convert_To_Intcost)

    graph.add_edge(startnode, endingnode, Convert_To_Intcost, False)
    showgraph()
    updatedropdown()


def enternode():
    startnode = str(e1.get())
    g.add_node(startnode)
    graph.addnode(startnode)
    Convert_To_Intheuristicnode1 = int(e5.get())
    graph.createhuristic(startnode, Convert_To_Intheuristicnode1)

    # graph.coordinates[startnode]=[Convert_To_Intx1,Convert_To_Inty1]
    showgraph()
    updatedropdown()


def updatedropdown():
    w.children["menu"].delete(0, "end")
    for v in graph.nodes:
        w.children["menu"].add_command(label=v, command=lambda veh=v: start.set(veh))


def animate(listofcolorsforeachframe):
    pos = nx.circular_layout(g)
    ims = []

    for i in range(len(listofcolorsforeachframe)):
        ims.append(
            (nx.draw_networkx_nodes(g, pos, node_color=listofcolorsforeachframe[i], cmap=plt.get_cmap("rainbow"),
                                    vmin=0.0, vmax=1.0),)
        )
    #plt.ion()
    im_ani = animation.ArtistAnimation(f, ims, interval=1000, repeat_delay=3000, repeat=False,
                                       blit=True)
    #display_animation(im_ani)
    im_ani.show()




def showgraph():
    x = nx.circular_layout(g)
    # cmap = plt.get_cmap("rainbow")
    f.clf()
    nx.draw(g, pos=x, node_size=600, with_labels=True, font_weight='5',  # cmap=cmap,
            font_size=10, node_color="blue")
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos=x, edge_labels=labels, label_pos=0.5)
    # plt.axis('off')
    canvas.draw()

def clearnxgraph():
    updatedropdown()
    g.remove_nodes_from(graph.nodes)
    f.clf()
    # nx.draw(g, graph.coordinates, node_size=100, with_labels=True, font_weight='5', font_size=10)
    # labels = nx.get_edge_attributes(g, 'weight')
    # nx.draw_networkx_edge_labels(g, pos=graph.coordinates, edge_labels=labels, label_pos=0.5)
    # plt.axis('off')
    canvas.draw()


# declare the window
def onClose():
    root.destroy()
    exit()
    # stops the main loop and interpreter


root = tkinter.Tk()
root.protocol("WM_DELETE_WINDOW", onClose)

# set window width and height
root.geometry("4000x5000")
# set window title
root.title("Artificial intelligence project")
# reating a label

Label(text="enter edge", bg='grey', font=('Calibri', 13)).pack()
errormess = tkinter.Label(text="please enter a huristic", bg='grey', font=('Calibri', 13))

node1 = tkinter.Label(root, text="first node")
node1.place(x=200, y=100)
node2 = tkinter.Label(root, text="second node")
node2.place(x=200, y=150)

addegde = tkinter.Button(root, text='add edge', width=25, command=lambda: [enteredge(), clear()])
addegde.place(x=1050, y=155)

addnode = tkinter.Button(root, text='add node', width=25, command=lambda: [enternode(), clear()])
addnode.place(x=1050, y=125)

e1 = tkinter.Entry(root)
e1.place(x=350, y=100)
e2 = tkinter.Entry(root)
e2.place(x=350, y=150)

e3 = tkinter.Entry(root)
e3.place(x=800, y=150)
e4 = tkinter.Entry(root)
e4.place(x=500, y=150)
e5 = tkinter.Entry(root)
e5.place(x=500, y=100)

costfrom1to2 = tkinter.Label(root, text="cost")
costfrom1to2.place(x=700, y=150)
heuristic = tkinter.Label(root, text="heuristic")
heuristic.place(x=520, y=50)
nodenamess = tkinter.Label(root, text="node names")
nodenamess.place(x=370, y=50)
# creating a button

x = tkinter.Entry(root)
x.place(x=900, y=50)
# goal.set(iwillkillsomeone[0])  # default value
# x = OptionMenu(root, goal, *iwillkillsomeone)
# def setg():
#    goal=[]


DFS = tkinter.Button(root, text='DFS', width=25, command=lambda: [cleariteration(), depth()])
DFS.place(x=1050, y=250)
UCS = tkinter.Button(root, text='UCS', width=25, command=lambda: [cleariteration(), uniform()])
UCS.place(x=1050, y=300)
DLS = tkinter.Button(root, text='DLS', width=25, command=lambda: [cleariteration(), depthlimited()])
DLS.place(x=1050, y=350)
ass = tkinter.Button(root, text='ass', width=25, command=lambda: [cleariteration(), astar()])
ass.place(x=1050, y=400)
gredy = tkinter.Button(root, text='greedy', width=25, command=lambda: [cleariteration(), greedy()])
gredy.place(x=1050, y=450)
IDS = tkinter.Button(root, text='IDS', width=25, command=lambda: [cleariteration(), iterative()])
IDS.place(x=1050, y=500)
BFS = tkinter.Button(root, text='BFS', width=25, command=lambda: [cleariteration(), breadth()])
BFS.place(x=1050, y=550)
cleargraphn = tkinter.Button(root, text='clear graph', width=25,
                             command=lambda: [clearnxgraph(), graph.cleargraph(), updatedropdown(), cleariteration()])
cleargraphn.place(x=1050, y=650)


graph.add_edge('A', 'B', 4, False)
graph.add_edge('B', 'D', 6, False)
graph.add_edge('B', 'E', 45, False)
graph.add_edge('C', 'C', 39, False)
graph.add_edge('C', 'D', 19, False)
graph.add_edge('C', 'F', 41, False)
graph.add_edge('D', 'C', 44, False)
graph.add_edge('D', 'E', 34, False)
graph.add_edge('E', 'G', 3, False)
graph.add_edge('F', 'G', 31, False)
graph.add_edge('G', 'E', 4, False)
graph.add_edge('G', 'H', 3, False)
graph.add_edge('H', 'I', 31, False)
graph.createhuristic("A", 12)
graph.createhuristic("B", 3)
graph.createhuristic("C", 4)
graph.createhuristic("D", 55)
graph.createhuristic("E", 44)
graph.createhuristic("F", 3)
graph.createhuristic("G", 34)
graph.createhuristic("H", 4)
graph.createhuristic("I", 44)
g.add_edge('A', 'B', weight=4)
g.add_edge('B', 'D', weight=6)
g.add_edge('B', 'E', weight=45)
g.add_edge('C', 'C', weight=39)
g.add_edge('C', 'D', weight=19)
g.add_edge('C', 'F', weight=41)
g.add_edge('D', 'C', weight=44)
g.add_edge('D', 'E', weight=34)
g.add_edge('E', 'G', weight=3)
g.add_edge('F', 'G', weight=31)
g.add_edge('G', 'E', weight=4)
g.add_edge('G', 'H', weight=3)
g.add_edge('H', 'I', weight=31)

global start
start = StringVar(root)
allnodesingraph = list(graph.nodes)
start.set(allnodesingraph[0])  # default value
w = OptionMenu(root, start, *allnodesingraph)
w.place(x=800, y=50)

# set window background color
root.configure(bg='grey')

global traversal
traversal = tkinter.Text(root, height=20, width=50)
traversal.place(x=600, y=250)
traversal.insert(tkinter.END, "")


def clear():
    e1.delete(0, 'end')
    e2.delete(0, 'end')
    e3.delete(0, 'end')
    e4.delete(0, 'end')
    e5.delete(0, 'end')
    x.delete(0, "end")


def cleariteration():
    traversal.delete("1.0", "end")


f = plt.figure(figsize=(5, 4), dpi=100)
# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.draw_idle()
canvas.get_tk_widget().place(x=50, y=200)
# axes
a = f.add_subplot(111)
showgraph()
plt.axis('on')

root.mainloop()
