import tkinter as tk
from tkinter import simpledialog, ttk, messagebox
import math
import time
import sys
import heapq

class ShortestPathGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shortest Path")

        # Create a canvas to draw the graph
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack(side=tk.LEFT, padx=20, pady=20)

        # Create a frame for UI elements
        self.control_frame = tk.Frame(root, bg="white")
        self.control_frame.pack(side=tk.LEFT)

        # Create a frame for the result and working text areas
        self.text_frame = tk.Frame(root)
        self.text_frame.pack(side=tk.LEFT, padx=20, pady=20)

        # Initialize dictionaries to store nodes and edges
        self.nodes = {}
        self.edges = {}

        # Create a "Draw Graph" button
        self.draw_button = tk.Button(self.control_frame, text="Draw Graph", command=self.draw_graph, bg="#4CAF50", fg="white", padx=10, pady=5, cursor="hand2")
        self.draw_button.pack()

        # Create a label for selecting the algorithm
        self.algorithm_label = tk.Label(self.control_frame, text="Select Algorithm:", bg="white")
        self.algorithm_label.pack()

        # Create a dropdown menu for selecting the algorithm
        self.algorithm_var = tk.StringVar(value="A*")
        self.algorithm_menu = ttk.Combobox(self.control_frame, textvariable=self.algorithm_var, values=["Bellman-Ford", "Dijkstra", "A*"])
        self.algorithm_menu.pack()

        # Create a "Find Shortest Path" button
        self.find_button = tk.Button(self.control_frame, text="Find Shortest Path", command=self.find_shortest_path, bg="#4CAF50", fg="white", padx=10, pady=5, cursor="hand2")
        self.find_button.pack()

        # Initialize source and destination nodes
        self.source_node = None
        self.destination_node = None

        # Create a frame for the result text area
        self.result_frame = tk.Frame(self.text_frame, bg="white")
        self.result_frame.pack(side=tk.TOP)

        # Create a text area for displaying results
        self.result_text = tk.Text(self.result_frame, height=10, width=50)
        self.result_text.pack()

        # Create a frame for the working text area
        self.working_frame = tk.Frame(self.text_frame, bg="white")
        self.working_frame.pack(side=tk.TOP)

        # Create a text area for displaying step-by-step working
        self.working_text = tk.Text(self.working_frame, height=10, width=50)
        self.working_text.pack(side=tk.LEFT)

        # Create a scrollbar for the working text area
        self.working_scrollbar = tk.Scrollbar(self.working_frame, command=self.working_text.yview)
        self.working_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.working_text.config(yscrollcommand=self.working_scrollbar.set)

    def draw_graph(self):
        # Clear the canvas and reset node and edge dictionaries
        self.clear_canvas()

        # Prompt user for the number of nodes
        num_nodes = simpledialog.askinteger("Number of Nodes", "Enter the number of nodes:")
        if num_nodes is None:
            return

        # Create nodes and collect their coordinates
        for i in range(1, num_nodes + 1):
            x, y = self.get_node_coordinates(i, num_nodes)
            self.nodes[i] = (x, y)
            self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")
            self.canvas.create_text(x, y - 20, text=str(i))

        # Prompt user for edge weights and create edges
        for i in range(1, num_nodes + 1):
            for j in range(i + 1, num_nodes + 1):
                response = simpledialog.askstring(
                    f"Edge {i}-{j}",
                    f"Enter weight for edge {i}-{j} (or leave empty for no edge):"
                )
                if response:
                    weight = int(response)
                    self.edges[(i, j)] = weight
                    x1, y1 = self.nodes[i]
                    x2, y2 = self.nodes[j]
                    self.canvas.create_line(x1, y1, x2, y2, fill="black")
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(weight))

        # Prompt user for source and destination nodes
        self.source_node = simpledialog.askinteger("Source Node", "Enter the source node:", minvalue=1, maxvalue=num_nodes)
        self.destination_node = simpledialog.askinteger("Destination Node", "Enter the destination node:", minvalue=1, maxvalue=num_nodes)

    def find_shortest_path(self):
        # Check if source and destination nodes are specified
        if self.source_node is None or self.destination_node is None:
            tk.messagebox.showerror("Error", "Source and destination nodes must be specified.")
            return

        # Get the selected algorithm
        algorithm = self.algorithm_var.get()

        # Record the start time for measuring execution time
        start_time = time.time()

        # Find the shortest path using the selected algorithm
        if algorithm == "Bellman-Ford":
            result, working_steps = self.find_shortest_path_bellman_ford()
            algorithm_name = "Bellman-Ford"
        elif algorithm == "Dijkstra":
            result, working_steps = self.find_shortest_path_dijkstra()
            algorithm_name = "Dijkstra"
        elif algorithm == "A*":
            result, working_steps = self.find_shortest_path_astar()
            algorithm_name = "A*"

        # Record the end time and calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Calculate space efficiency
        space_efficiency = sys.getsizeof(self.edges) + sys.getsizeof(self.nodes)

        # Display the result in the result text area
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"The shortest path from node {self.source_node} to node {self.destination_node} is: {result['path']} with a distance of {result['distance']}\n"
                                          f"Execution Time ({algorithm_name}): {execution_time:.6f} seconds\n"
                                          f"Space Efficiency ({algorithm_name}): {space_efficiency} bytes\n"
                                          f"Algorithm Comparison: {self.compare_algorithms()}\n")

        # Display the step-by-step working in the working text area
        self.working_text.delete(1.0, tk.END)
        self.working_text.insert(tk.END, working_steps)

        # Highlight the shortest path on the canvas
        self.highlight_path(result['path'])

    def find_shortest_path_bellman_ford(self):
        # Implement Bellman-Ford algorithm to find the shortest path
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        INF = float("inf")

        dist = [INF] * (num_nodes + 1)
        pred = [-1] * (num_nodes + 1)
        dist[self.source_node] = 0

        working_steps = "Bellman-Ford Algorithm Working Steps:\n\n"

        for _ in range(num_nodes - 1):
            for (u, v), weight in self.edges.items():
                if dist[u] != INF and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    pred[v] = u

                    # Append working step to the working_text
                    working_steps += f"Step: Relax edge ({u}, {v}), New distance to node {v} = {dist[v]}\n"

        for (u, v), weight in self.edges.items():
            if dist[u] != INF and dist[u] + weight < dist[v]:
                tk.messagebox.showerror("Error", "Negative cycle detected. Bellman-Ford does not work in this case.")
                return {"path": [], "distance": float("inf")}, working_steps

        path = [self.destination_node]
        current_node = self.destination_node
        while current_node != self.source_node:
            current_node = pred[current_node]
            path.append(current_node)
        path.reverse()

        return {"path": path, "distance": dist[self.destination_node]}, working_steps

    def find_shortest_path_dijkstra(self):
        # Implement Dijkstra's algorithm to find the shortest path
        num_nodes = len(self.nodes)
        INF = float("inf")

        dist = [INF] * (num_nodes + 1)
        pred = [-1] * (num_nodes + 1)
        visited = [False] * (num_nodes + 1)
        dist[self.source_node] = 0

        working_steps = "Dijkstra's Algorithm Working Steps:\n\n"

        for _ in range(num_nodes):
            u = self.min_distance(dist, visited)
            visited[u] = True

            for v in range(1, num_nodes + 1):
                if not visited[v] and self.edges.get((u, v)) is not None:
                    if dist[u] + self.edges[(u, v)] < dist[v]:
                        dist[v] = dist[u] + self.edges[(u, v)]
                        pred[v] = u

                        # Append working step to the working_text
                        working_steps += f"Step: Relax edge ({u}, {v}), New distance to node {v} = {dist[v]}\n"

        path = [self.destination_node]
        current_node = self.destination_node
        while current_node != self.source_node:
            current_node = pred[current_node]
            path.append(current_node)
        path.reverse()

        return {"path": path, "distance": dist[self.destination_node]}, working_steps

    def min_distance(self, dist, visited):
        # Helper function to find the node with the minimum distance
        min_dist = float("inf")
        min_index = -1
        for v, d in enumerate(dist):
            if not visited[v] and d < min_dist:
                min_dist = d
                min_index = v
        return min_index

    def find_shortest_path_astar(self):
        # Implement A* algorithm to find the shortest path
        num_nodes = len(self.nodes)
        INF = float("inf")

        # Define a heuristic function for A* (Euclidean distance to the destination node)
        def heuristic(node1, node2):
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        open_set = [(0 + heuristic(self.source_node, self.destination_node), self.source_node)]
        heapq.heapify(open_set)
        g_score = [INF] * (num_nodes + 1)
        g_score[self.source_node] = 0
        came_from = {}

        working_steps = "A* Algorithm Working Steps:\n\n"

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == self.destination_node:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return {"path": path, "distance": g_score[self.destination_node]}, working_steps

            for neighbor in range(1, num_nodes + 1):
                if self.edges.get((current, neighbor)) is not None:
                    tentative_g_score = g_score[current] + self.edges[(current, neighbor)]

                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, self.destination_node)

                        # Append working step to the working_text
                        working_steps += f"Step: Consider neighbor {neighbor}, New distance to node {neighbor} = {g_score[neighbor]}\n"

                        if neighbor not in [node[1] for node in open_set]:
                            heapq.heappush(open_set, (f_score, neighbor))

        return {"path": [], "distance": float("inf")}, working_steps

    def clear_canvas(self):
        # Clear the canvas and reset node and edge dictionaries
        self.canvas.delete("all")
        self.nodes = {}
        self.edges = {}

    def get_node_coordinates(self, i, num_nodes):
        # Calculate the coordinates for node placement in a circle
        angle = (2 * math.pi * (i - 1)) / num_nodes
        x = 300 + 250 * math.cos(angle)
        y = 300 + 250 * math.sin(angle)
        return x, y

    def highlight_path(self, path):
        # Highlight the shortest path on the canvas
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if v in self.nodes:
                x1, y1 = self.nodes[u]
                x2, y2 = self.nodes[v]
                self.canvas.create_line(x1, y1, x2, y2, fill="green", width=3)

    def compare_algorithms(self):
        # Compare Bellman-Ford, Dijkstra, and A* algorithms based on execution time and space efficiency
        bellman_ford_time = 0
        dijkstra_time = 0
        astar_time = 0
        bellman_ford_space = 0
        dijkstra_space = 0
        astar_space = 0
        num_trials = 10

        for _ in range(num_trials):
            start_time = time.time()
            self.find_shortest_path_bellman_ford()
            end_time = time.time()
            bellman_ford_time += end_time - start_time

            start_time = time.time()
            self.find_shortest_path_dijkstra()
            end_time = time.time()
            dijkstra_time += end_time - start_time

            start_time = time.time()
            self.find_shortest_path_astar()
            end_time = time.time()
            astar_time += end_time - start_time

            bellman_ford_space += sys.getsizeof(self.edges) + sys.getsizeof(self.nodes)
            dijkstra_space += sys.getsizeof(self.edges) + sys.getsizeof(self.nodes)
            astar_space += sys.getsizeof(self.edges) + sys.getsizeof(self.nodes)

        bellman_ford_time /= num_trials
        dijkstra_time /= num_trials
        astar_time /= num_trials
        bellman_ford_space /= num_trials
        dijkstra_space /= num_trials
        astar_space /= num_trials

        if bellman_ford_time < dijkstra_time and bellman_ford_time < astar_time and bellman_ford_space < dijkstra_space and bellman_ford_space < astar_space:
            return "Bellman-Ford is better (based on both time and space efficiency)"
        elif dijkstra_time < bellman_ford_time and dijkstra_time < astar_time and dijkstra_space < bellman_ford_space and dijkstra_space < astar_space:
            return "Dijkstra is better (based on both time and space efficiency)"
        elif astar_time < bellman_ford_time and astar_time < dijkstra_time and astar_space < bellman_ford_space and astar_space < dijkstra_space:
            return "A* is better (based on both time and space efficiency)"
        else:
            return "All algorithms perform similarly"

if __name__ == "__main__":
    root = tk.Tk()
    app = ShortestPathGUI(root)
    root.mainloop()
