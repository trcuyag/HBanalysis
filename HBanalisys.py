import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, simpledialog
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import networkx as nx

print("1. Importaciones OK")

class HBfindApp:
    def __init__(self, root):
        print("2. Iniciando __init__")
        self.root = root
        root.title("HBanalysis - GUI")

        # Rutas de archivos
        self.xpm_path = tk.StringVar()
        self.log_path = tk.StringVar()
        self.out_path = tk.StringVar()
        self.xvg_path = tk.StringVar()
        self.ndx_path = tk.StringVar()
        self.pdb_path = tk.StringVar()

        # Datos de la simulación
        self.nframes = 0
        self.nlines = 0
        self.matrix = []          # Lista de strings (cada string = fila del XPM)
        self.donors = []
        self.hydrogens = []
        self.acceptors = []
        self.counts_o = []
        self.counts_dash = []
        self.counts_star = []
        self.prevalences = []
        self.calculated = False

        self.build_widgets()
        print("3. Widgets construidos")

    def build_widgets(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = tk.LabelFrame(main_frame, text="Files", padx=10, pady=10)
        file_frame.pack(fill=tk.X, pady=5)

        def add_row(label, var, browse_cmd):
            row = tk.Frame(file_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, width=12, anchor="w").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=60).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            tk.Button(row, text="Search", command=browse_cmd).pack(side=tk.LEFT)

        add_row(".xpm:", self.xpm_path, self.browse_xpm)
        add_row(".log:", self.log_path, self.browse_log)
        add_row("Salida:", self.out_path, self.browse_out)
        add_row(".xvg:", self.xvg_path, self.browse_xvg)
        add_row(".ndx:", self.ndx_path, self.browse_ndx)
        add_row(".pdb:", self.pdb_path, self.browse_pdb)

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="▶ Calculate", command=self.calculate, bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="📊 Correlation", command=self.compute_correlation, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🌐 Network Analysis", command=self.analyze_network, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clean", command=lambda: self.output_text.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=5)

        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = tk.Text(text_frame, font=("Courier", 9), wrap=tk.NONE)
        scrollbar_y = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar_x = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.output_text.xview)
        self.output_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        xvg_frame = tk.LabelFrame(main_frame, text="Export graph (.xvg)", padx=10, pady=10)
        xvg_frame.pack(fill=tk.X, pady=5)

        tk.Label(xvg_frame, text="Nº of Hbond:").pack(side=tk.LEFT)
        self.bond_entry = tk.Entry(xvg_frame, width=10)
        self.bond_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(xvg_frame, text="Generate .xvg", command=self.generate_xvg, bg="lightgreen").pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        tk.Label(main_frame, textvariable=self.status_var, fg="gray").pack(pady=5)

        citation_frame = tk.LabelFrame(main_frame, text="Cite", padx=10, pady=10, fg="blue")
        citation_frame.pack(fill=tk.X, pady=5)
        citation_text = (
            "Author: Teobaldo Cuya\n"
            "Department of Mathematics, Physics and Computing\n"
            "Faculty of Technology, Universidade Estadual de Rio de Janeiro\n"
            "Laboratorio de Computação Avançada em Modelagem Molecular - LCAMM"
        )
        tk.Label(citation_frame, text=citation_text, justify=tk.LEFT, font=("Arial", 9, "italic"), fg="navy").pack(anchor="w")

    # ----- Funciones de navegación -----
    def browse_xpm(self):
        f = filedialog.askopenfilename(filetypes=[("XPM files", "*.xpm"), ("All", "*.*")])
        if f: self.xpm_path.set(f)

    def browse_log(self):
        f = filedialog.askopenfilename(filetypes=[("Log files", "*.log"), ("All", "*.*")])
        if f: self.log_path.set(f)

    def browse_out(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if f: self.out_path.set(f)

    def browse_xvg(self):
        f = filedialog.asksaveasfilename(defaultextension=".xvg", filetypes=[("XVG", "*.xvg")])
        if f: self.xvg_path.set(f)

    def browse_ndx(self):
        f = filedialog.askopenfilename(filetypes=[("Index files", "*.ndx"), ("All", "*.*")])
        if f: self.ndx_path.set(f)

    def browse_pdb(self):
        f = filedialog.askopenfilename(filetypes=[("PDB files", "*.pdb"), ("All", "*.*")])
        if f: self.pdb_path.set(f)

    # ----- Parseadores -----
    def parse_xpm(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('s'):
                start_idx = i
                break
        if start_idx == -1:
            raise ValueError("No se encontró 'static char'")
        header_line = lines[start_idx + 1]
        nums = re.findall(r'\d+', header_line)
        if len(nums) < 4:
            raise ValueError("Formato XPM incorrecto")
        nframes = int(nums[0])
        nlines = int(nums[1])
        idx = start_idx + 2
        while idx < len(lines) and not lines[idx].strip().startswith('/* y-axis:'):
            idx += 1
        if idx >= len(lines):
            raise ValueError("No se encontró '/* y-axis:'")
        idx += 1
        while idx < len(lines) and not lines[idx].strip().startswith('"'):
            idx += 1
        if idx >= len(lines):
            raise ValueError("No se encontró el inicio de los datos")
        matrix = []
        for i in range(nlines):
            if idx + i >= len(lines):
                break
            raw = lines[idx + i].strip()
            raw = raw.strip().strip('"').rstrip(',')
            matrix.append(raw)
        matrix = matrix[::-1]  # Invertir para que matrix[0] sea el primer enlace
        if len(matrix) < nlines:
            print(f"Advertencia: solo se leyeron {len(matrix)} líneas de {nlines}")
            nlines = len(matrix)
        return nframes, nlines, matrix

    def parse_log(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        donors, hydrogens, acceptors = [], [], []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                donors.append(parts[0])
                hydrogens.append(parts[1])
                acceptors.append(parts[2])
        return donors, hydrogens, acceptors

    # ----- Calcular -----
    def calculate(self):
        if not os.path.exists(self.xpm_path.get()):
            messagebox.showerror("Error", "Falta .xpm")
            return
        if not os.path.exists(self.log_path.get()):
            messagebox.showerror("Error", "Falta .log")
            return
        if not self.out_path.get():
            messagebox.showerror("Error", "Falta archivo de salida")
            return
        try:
            self.status_var.set("Reading...")
            self.root.update()
            self.nframes, self.nlines, self.matrix = self.parse_xpm(self.xpm_path.get())
            self.donors, self.hydrogens, self.acceptors = self.parse_log(self.log_path.get())
            if self.nlines != len(self.donors):
                messagebox.showwarning("Warning",
                    f"Número de enlaces en .log ({len(self.donors)}) no coincide con .xpm ({self.nlines}). Se usará el menor.")
                self.nlines = min(self.nlines, len(self.donors))
            self.counts_o = [0] * self.nlines
            self.counts_dash = [0] * self.nlines
            self.counts_star = [0] * self.nlines
            self.prevalences = [0.0] * self.nlines
            for i in range(self.nlines):
                line = self.matrix[i]
                self.counts_o[i] = line.count('o')
                self.counts_dash[i] = line.count('-')
                self.counts_star[i] = line.count('*')
                self.prevalences[i] = (self.counts_o[i] / self.nframes) * 100.0
            out_lines = []
            header = f"{'bond':<6} {'donor':<12} {'acceptor':<12} {'count':>8} {'%':>8} {'#ins':>8} {'#ins+pres':>10}"
            out_lines.append(header)
            out_lines.append("-" * 70)
            for i in range(self.nlines):
                out_lines.append(
                    f"{i+1:4d})  {self.donors[i]:<12} {self.acceptors[i]:<12} "
                    f"{self.counts_o[i]:8d} {self.prevalences[i]:8.2f}  {self.counts_dash[i]:8d} {self.counts_star[i]:10d}"
                )
            output_str = "\n".join(out_lines)
            self.output_text.delete('1.0', tk.END)
            self.output_text.insert(tk.END, output_str)
            with open(self.out_path.get(), 'w') as f:
                f.write(output_str)
            self.calculated = True
            self.status_var.set(f"Completado. {self.nlines} enlaces.")
            messagebox.showinfo("Success", f"Table save in {self.out_path.get()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error in calculations.")

    # ----- Correlación (con zoom y guardado) -----
    def compute_correlation(self):
        if not self.calculated:
            messagebox.showerror("Error", "Primero calcula prevalencia.")
            return
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError as e:
            messagebox.showerror("Error", f"Missing libraries: {e}\nInstala: pip install numpy matplotlib")
            return
        try:
            self.status_var.set("Calculating correlation...")
            self.root.update()
            n = self.nlines
            m = self.nframes
            data = np.zeros((n, m), dtype=int)
            for i in range(n):
                line = self.matrix[i]
                for j, ch in enumerate(line[:m]):
                    if ch == 'o':
                        data[i, j] = 1
            corr = np.corrcoef(data)
            corr = np.nan_to_num(corr)
            win = tk.Toplevel(self.root)
            win.title("Correlation Matrix")
            win.geometry("800x800")  # Aumentado verticalmente
            fig, ax = plt.subplots(figsize=(9, 7))
            cmap = LinearSegmentedColormap.from_list('div', ['blue', 'white', 'red'])
            im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='auto', interpolation='none')
            ax.set_xlabel("Bond")
            ax.set_ylabel("Bond")
            ax.set_title("Pearson correlation")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Correlation")
            step = max(1, n // 20)
            ax.set_xticks(range(0, n, step))
            ax.set_xticklabels([str(i+1) for i in range(0, n, step)])
            ax.set_yticks(range(0, n, step))
            ax.set_yticklabels([str(i+1) for i in range(0, n, step)])
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()

            def save_corr():
                file = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
                if file:
                    fig.savefig(file, dpi=1200, format='jpg' if file.endswith('.jpg') else 'png', bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image save in {file}")
            # Frame para botones en la parte inferior
            btn_frame_corr = tk.Frame(win)
            btn_frame_corr.pack(pady=5)
            tk.Button(btn_frame_corr, text="Save image", command=save_corr).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame_corr, text="Close", command=win.destroy).pack(side=tk.LEFT, padx=5)

            self.status_var.set("Correlation showed.")
        except Exception as e:
            messagebox.showerror("Error", f"No correlation could be calculated:\n{e}")
            self.status_var.set("Error in correlation.")

    # ----- Generar .xvg -----
    def generate_xvg(self):
        if not self.calculated:
            messagebox.showerror("Error", "Calculate prevalence first.")
            return
        idx_str = self.bond_entry.get().strip()
        if not idx_str:
            messagebox.showerror("Error", "Enter a number.")
            return
        try:
            j = int(idx_str)
        except ValueError:
            messagebox.showerror("Error", "Must be integer.")
            return
        if j < 1 or j > self.nlines:
            messagebox.showerror("Error", f"Enlace entre 1 y {self.nlines}")
            return
        default_label = f"{self.donors[j-1]}-{self.hydrogens[j-1]}-{self.acceptors[j-1]}"
        label = simpledialog.askstring("Etiqueta", "Etiqueta:", initialvalue=default_label)
        if not label:
            label = default_label
        line = self.matrix[j-1]
        xvg_path = self.xvg_path.get()
        if not xvg_path:
            messagebox.showerror("Error", "Especifica ruta de .xvg")
            return
        try:
            with open(xvg_path, 'w') as f:
                f.write(f"# {label} = {self.donors[j-1]} {self.hydrogens[j-1]} {self.acceptors[j-1]}\n")
                for idx_char, ch in enumerate(line):
                    if ch == 'o':
                        f.write(f"{idx_char + 2} {label}\n")
            self.status_var.set(f".xvg generado: {xvg_path}")
            messagebox.showinfo("Success", f"Saved in {xvg_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ================= ANÁLISIS DE RED =================
    def analyze_network(self):
        if not self.calculated:
            messagebox.showerror("Error", "You must first execute 'Calculation' to have the occupancy matrix.")
            return
        if not os.path.exists(self.ndx_path.get()):
            messagebox.showerror("Error", "Falta el archivo .ndx")
            return
        if not os.path.exists(self.pdb_path.get()):
            messagebox.showerror("Error", "Falta el archivo .pdb")
            return

        try:
            self.status_var.set("Analysing network...")
            self.root.update()

            # 1. Parsear .ndx para obtener triples donor-hydrogen-acceptor
            hbonds_triples = self.parse_ndx_triples(self.ndx_path.get())
            if not hbonds_triples:
                messagebox.showerror("Error", "No triple links were found in the file .ndx")
                return
            if len(hbonds_triples) != self.nlines:
                messagebox.showwarning("Warning",
                    f"The number of bonds in .ndx ({len(hbonds_triples)}) does not match .xpm ({self.nlines}). "
                    "Se usarán los primeros {min} enlaces.")
                hbonds_triples = hbonds_triples[:self.nlines]
                if len(hbonds_triples) == 0:
                    messagebox.showerror("Error", "No hay suficientes triples.")
                    return

            # 2. Parsear .pdb para obtener mapa átomo -> (residuo, nombre_residuo)
            atom_to_residue = self.parse_pdb_residues(self.pdb_path.get())

            # 3. Construir grafo y también guardar la lista de (residuo_donor, residuo_aceptor) por enlace
            G = nx.Graph()
            edge_weights = {}          # clave: (res1, res2) -> peso (fracción)
            bond_residue_pairs = []    # lista de (res_donor, res_aceptor) para cada enlace (índice)

            for idx, (donor_atom, hydrogen_atom, acceptor_atom) in enumerate(hbonds_triples):
                if idx >= self.nlines:
                    break
                donor_res = atom_to_residue.get(donor_atom)
                acc_res = atom_to_residue.get(acceptor_atom)
                if donor_res is None or acc_res is None:
                    print(f"Warning: bondf {idx+1} could not be mapped to residue (donor={donor_atom}, acc={acceptor_atom})")
                    bond_residue_pairs.append((None, None))
                    continue
                if donor_res == acc_res:
                    bond_residue_pairs.append((donor_res, acc_res))
                    continue
                weight = self.prevalences[idx] / 100.0
                edge = tuple(sorted([donor_res, acc_res]))
                if edge in edge_weights:
                    edge_weights[edge] += weight
                else:
                    edge_weights[edge] = weight
                bond_residue_pairs.append((donor_res, acc_res))

            for (r1, r2), w in edge_weights.items():
                G.add_node(r1, label=f"{r1[1]}{r1[0]}")
                G.add_node(r2, label=f"{r2[1]}{r2[0]}")
                G.add_edge(r1, r2, weight=w)

            if G.number_of_nodes() == 0:
                messagebox.showinfo("Red vacía", "No se encontraron enlaces entre residuos diferentes.")
                self.status_var.set("Red vacía.")
                return

            # 4. Calcular métricas de centralidad
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G, weight='weight')
            closeness_cent = nx.closeness_centrality(G, distance='weight')
            eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

            # 5. Calcular ocupación por residuo (fracción de frames con al menos un enlace activo)
            residue_occupancy = self.compute_residue_occupancy(bond_residue_pairs)

            # 6. Mostrar resultados en el área de texto
            self.output_text.insert(tk.END, "\n" + "="*70 + "\n")
            self.output_text.insert(tk.END, "ANALYSIS OF HYDROGEN BOND NETWORKS\n")
            self.output_text.insert(tk.END, "="*70 + "\n\n")
            self.output_text.insert(tk.END, f"Number of nodes (residuos): {G.number_of_nodes()}\n")
            self.output_text.insert(tk.END, f"Number of edges (links): {G.number_of_edges()}\n\n")

            # Tabla de ocupación por residuo (ordenada descendente)
            self.output_text.insert(tk.END, "RESIDUE OCCUPANCY (fraction of frames with at least one active link):\n")
            self.output_text.insert(tk.END, f"{'Residuo':<12} {'Ocupación':>12} {'#Enlaces':>10}\n")
            self.output_text.insert(tk.END, "-"*40 + "\n")
            sorted_residues = sorted(residue_occupancy.items(), key=lambda x: x[1]['occupancy'], reverse=True)
            for (resnum, resname), data in sorted_residues:
                label = f"{resname}{resnum}"
                occ = data['occupancy'] * 100.0
                n_edges = data['n_edges']
                self.output_text.insert(tk.END, f"{label:<12} {occ:>11.2f}% {n_edges:>10d}\n")
            self.output_text.insert(tk.END, "\n")

            # Tabla de centralidades
            self.output_text.insert(tk.END, "Centrality metrics per node\n")
            self.output_text.insert(tk.END, f"{'Residuo':<12} {'Grado':>8} {'Intermed.':>12} {'Cercanía':>12} {'Vector propio':>14}\n")
            self.output_text.insert(tk.END, "-"*70 + "\n")
            for node in sorted(G.nodes(), key=lambda x: (x[1], x[0])):
                label = f"{node[1]}{node[0]}"
                deg = degree_cent.get(node, 0)
                bet = betweenness_cent.get(node, 0)
                clo = closeness_cent.get(node, 0)
                eig = eigenvector_cent.get(node, 0)
                self.output_text.insert(tk.END, f"{label:<12} {deg:8.4f} {bet:12.4f} {clo:12.4f} {eig:14.4f}\n")

            # Aristas con mayor peso
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            self.output_text.insert(tk.END, "\nEdges with greater weight (ocupación):\n")
            for u, v, data in sorted_edges[:10]:
                self.output_text.insert(tk.END, f"  {u[1]}{u[0]} -- {v[1]}{v[0]}  (peso = {data['weight']:.3f})\n")

            self.output_text.see(tk.END)

            # 7. Visualizar la red (ventana más alta)
            self.show_network_graph(G, degree_cent, betweenness_cent, closeness_cent, eigenvector_cent)

            self.status_var.set("Network analysis completed.")
            messagebox.showinfo("Success", "Network analysis completed.")

        except Exception as e:
            messagebox.showerror("Error", f"Error in the Network analysis:\n{str(e)}")
            self.status_var.set("Error in the Network analysis.")
            import traceback
            traceback.print_exc()

    # ----- Función para calcular ocupación por residuo -----
    def compute_residue_occupancy(self, bond_residue_pairs):
        """
        Calcula, para cada residuo, la fracción de frames en los que al menos uno
        de sus enlaces de H está activo.
        bond_residue_pairs: lista de tuplas (res_donor, res_aceptor) por enlace.
        """
        # Mapear residuo -> lista de índices de enlace en los que participa (donde el residuo no es None)
        residue_to_bonds = {}
        for idx, (d, a) in enumerate(bond_residue_pairs):
            if d is not None and d != a:
                residue_to_bonds.setdefault(d, []).append(idx)
            if a is not None and a != d:
                residue_to_bonds.setdefault(a, []).append(idx)
            # Si d == a, el enlace es intramolecular (no se considera para ocupación de residuo)

        # Para cada residuo, calcular ocupación
        occupancy = {}
        for residue, bond_indices in residue_to_bonds.items():
            # bond_indices es una lista de índices de enlace
            # Para cada frame, ver si al menos uno de esos enlaces tiene 'o'
            # Usamos la matriz self.matrix (filas = enlaces, columnas = frames)
            n_frames = self.nframes
            count_on = 0
            for frame in range(n_frames):
                # Verificar si algún enlace de este residuo está activo en este frame
                for bond_idx in bond_indices:
                    if bond_idx < len(self.matrix) and frame < len(self.matrix[bond_idx]):
                        if self.matrix[bond_idx][frame] == 'o':
                            count_on += 1
                            break  # ya encontramos uno activo, pasamos al siguiente frame
            occupancy[residue] = {
                'occupancy': count_on / n_frames if n_frames > 0 else 0,
                'n_edges': len(bond_indices)
            }
        return occupancy

    # ----- Funciones auxiliares para parseo -----
    def parse_ndx_triples(self, ndx_path):
        with open(ndx_path, 'r') as f:
            lines = f.readlines()

        sections = {}
        current_section = None
        current_numbers = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                if current_section is not None and current_numbers:
                    sections[current_section] = current_numbers
                current_section = line[1:-1].strip()
                current_numbers = []
            else:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    current_numbers.extend([int(n) for n in numbers])
        if current_section is not None and current_numbers:
            sections[current_section] = current_numbers

        target_section = None
        for sec in sections:
            if 'hbonds' in sec.lower():
                target_section = sec
                break
        if target_section is None:
            for sec in reversed(list(sections.keys())):
                if len(sections[sec]) % 3 == 0 and len(sections[sec]) >= 3:
                    target_section = sec
                    break
        if target_section is None:
            return []

        numbers = sections[target_section]
        triples = [(numbers[i], numbers[i+1], numbers[i+2]) for i in range(0, len(numbers)-2, 3)]
        return triples

    def parse_pdb_residues(self, pdb_path):
        atom_to_res = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        atom_serial = int(parts[1])
                    except:
                        continue
                    if len(parts) >= 7 and parts[4].isalpha() and len(parts[4]) == 1:
                        resname = parts[3]
                        resnum = int(parts[5])
                    else:
                        resname = parts[3]
                        resnum = int(parts[4])
                    atom_to_res[atom_serial] = (resnum, resname)
        return atom_to_res

    # ================= VISUALIZACIÓN CON RESALTADO Y VENTANA MÁS ALTA =================
    def show_network_graph(self, G, degree_cent, betweenness_cent, closeness_cent, eigenvector_cent):
        """Muestra el grafo en una ventana Tkinter con barra de herramientas (zoom, desplazamiento),
        botón para guardar imagen en JPG a 1200 dpi, y una caja para resaltar un residuo específico.
        La ventana se ha hecho más alta (800x800) para que los botones sean visibles.
        """
        try:
            win = tk.Toplevel(self.root)
            win.title("Hydrogen bond network")
            win.geometry("800x800")  # Aumentado verticalmente

            fig, ax = plt.subplots(figsize=(9, 8))
            pos = nx.spring_layout(G, seed=42, k=0.5)

            # Dibujar nodos con colores y tamaños base
            node_colors = [degree_cent.get(node, 0) for node in G.nodes()]
            node_sizes = [100 + 500 * betweenness_cent.get(node, 0) for node in G.nodes()]

            # Guardar referencia a los nodos dibujados para poder actualizar después
            nodes_drawn = nx.draw_networkx_nodes(G, pos, ax=ax,
                                                 node_color=node_colors,
                                                 cmap=plt.cm.viridis,
                                                 node_size=node_sizes,
                                                 alpha=0.9,
                                                 label="Nodos")

            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_w = max(edge_weights) if edge_weights else 1
            edge_widths = [1 + 4 * (w / max_w) for w in edge_weights]
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.6, edge_color='gray')

            labels = {node: f"{node[1]}{node[0]}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)

            ax.set_title("Hbonds Network (coulor = grade, size = intermediation)")
            ax.axis('off')

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('Degree centrality')

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()

            # --- Funciones para resaltar un nodo ---
            def highlight_node(node_label):
                """Resalta el nodo con un color y tamaño diferente."""
                # Buscar el nodo por su etiqueta (ej. "ASP114")
                target_node = None
                for node in G.nodes():
                    if f"{node[1]}{node[0]}" == node_label:
                        target_node = node
                        break
                if target_node is None:
                    messagebox.showerror("Error", f"No se encontró el residuo '{node_label}' en la red.")
                    return

                # Redibujar todos los nodos (para resetear colores)
                ax.clear()
                # Volver a dibujar todo
                nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.6, edge_color='gray')
                # Dibujar nodos normales
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                       node_color=node_colors,
                                       cmap=plt.cm.viridis,
                                       node_size=node_sizes,
                                       alpha=0.9)
                # Dibujar el nodo resaltado
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                       nodelist=[target_node],
                                       node_color='red',
                                       node_size=node_sizes[list(G.nodes()).index(target_node)] * 3,
                                       alpha=1.0,
                                       edgecolors='black',
                                       linewidths=2)
                nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)
                ax.set_title(f"Red de enlaces de H - Resaltado: {node_label} (color = grado, tamaño = intermediación)")
                ax.axis('off')
                canvas.draw()

            # --- Función para guardar imagen ---
            def save_image():
                file = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
                )
                if file:
                    fig.savefig(file, dpi=1200, format='jpg' if file.endswith('.jpg') else 'png', bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Image saved in {file}")

            # --- Frame para el resaltado y botones ---
            control_frame = tk.Frame(win)
            control_frame.pack(pady=5, fill=tk.X)

            # Sub-frame para resaltado
            highlight_frame = tk.Frame(control_frame)
            highlight_frame.pack(side=tk.LEFT, padx=5)
            tk.Label(highlight_frame, text="Highlight residue:").pack(side=tk.LEFT, padx=5)
            entry_highlight = tk.Entry(highlight_frame, width=15)
            entry_highlight.pack(side=tk.LEFT, padx=5)
            btn_highlight = tk.Button(highlight_frame, text="Highlight", 
                                      command=lambda: highlight_node(entry_highlight.get().strip()))
            btn_highlight.pack(side=tk.LEFT, padx=5)

            # Sub-frame para botones de guardar y cerrar
            btn_frame = tk.Frame(control_frame)
            btn_frame.pack(side=tk.RIGHT, padx=5)
            btn_save = tk.Button(btn_frame, text="Save as JPG (1200 dpi)", command=save_image)
            btn_save.pack(side=tk.LEFT, padx=5)
            btn_close = tk.Button(btn_frame, text="Close", command=win.destroy)
            btn_close.pack(side=tk.LEFT, padx=5)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo mostrar la red:\n{str(e)}")
            self.status_var.set("Error en visualización.")


if __name__ == "__main__":
    print("4. Creando root...")
    root = tk.Tk()
    print("5. Root creado")
    app = HBfindApp(root)
    print("6. App iniciada, entrando a mainloop...")
    root.mainloop()
    print("7. Fin")
