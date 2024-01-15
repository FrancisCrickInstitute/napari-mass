import itertools
import os
import subprocess


class ExternalTspSolver:
    def __init__(self, tsp_params):
        self.params = tsp_params
        self.temp_path = tsp_params.get('temp_path', '')

    def solve_tsp_concorde(self, distance_matrix):
        process_path = self.params.get('concorde_path')
        tsp_filename = os.path.join(self.temp_path, 'distances.tsp')
        output_filename = os.path.join(self.temp_path, 'solution.txt')
        self.export_tsp(tsp_filename, distance_matrix)
        output = self.subprocess_run([process_path, '-x', '-o', f'{output_filename}', f'{tsp_filename}'], output_filename)
        result = self.parse_output(output, -1)
        os.remove(tsp_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        return result

    def solve_tsp_lkh(self, distance_matrix):
        process_path = self.params.get('lkh_path')
        par_filename = os.path.join(self.temp_path, 'tsp.par')
        tsp_filename = os.path.join(self.temp_path, 'distances.tsp')
        output_filename = os.path.join(self.temp_path, 'solution.txt')
        self.export_tsp(tsp_filename, distance_matrix)
        self.export_par(par_filename, tsp_filename, output_filename)
        output = self.subprocess_run([process_path, par_filename], output_filename)
        result = self.parse_output(output, -2)
        os.remove(par_filename)
        os.remove(tsp_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        return result

    def subprocess_run(self, cmd, output_filename, show_errors=False):
        output = ''
        if os.path.exists(cmd[0]):
            completed_process = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if completed_process.returncode != 0 and show_errors:
                print(completed_process.stdout)
                print('returncode:', completed_process.returncode)
            if os.path.exists(output_filename):
                with open(output_filename) as file:
                    output = file.read()
        return output

    def export_tsp(self, filename, distance_matrix):
        lines = [
            'NAME: Section_Similarity_Data',
            'TYPE: TSP',
            f'DIMENSION: {len(distance_matrix) + 1}',
            'EDGE_WEIGHT_TYPE: EXPLICIT',
            'EDGE_WEIGHT_FORMAT: UPPER_ROW',
            'NODE_COORD_TYPE: NO_COORDS',
            'DISPLAY_DATA_TYPE: NO_DISPLAY',
            'EDGE_WEIGHT_SECTION'
        ]

        distances = [0] * len(distance_matrix)  # dummy city
        for i, j in itertools.combinations(range(len(distance_matrix)), 2):
            distances.append(int(round(distance_matrix[i][j])))
        for distance in distances:
            lines.append(str(distance))
        lines.append('EOF')

        with open(filename, 'w') as file:
            file.write('\n'.join(lines) + '\n')

    def export_par(self, filename, tsp_filename, output_filename):
        lines = [
            f'PROBLEM_FILE = {tsp_filename}',
            f'OUTPUT_TOUR_FILE = {output_filename}',
            'PRECISION = 1'
        ]

        with open(filename, 'w') as file:
            file.write('\n'.join(lines) + '\n')

    def parse_output(self, output, offset):
        result = []
        items = output.split()
        if len(items) > 1:
            if 'TOUR_SECTION' in items:
                start = items.index('TOUR_SECTION') + 1
            else:
                start = 1
            for x0 in items[start:]:
                if x0.isnumeric():
                    x = int(x0) + offset
                    if x >= 0:
                        result.append(x)
        return result
