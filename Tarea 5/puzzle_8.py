import heapq
import numpy as np 

class puzzle_8:
    def __init__(self, tablero, padre=None, movimiento="", profundidad=0, costo=0):
        """
        Clase que representa un estado del 8-Puzzle.
        """
        self.tablero = np.array(tablero)
        self.padre = padre
        self.movimiento = movimiento
        self.profundidad = profundidad
        self.costo = costo
        self.posicion_vacia = tuple(map(int, np.where(self.tablero == 0)))

    def __lt__(self, otro):
        """
        Método para comparar nodos basado en la función de costo f(n) = g(n) + h(n)
        """
        return (self.costo + self.profundidad) < (otro.costo + otro.profundidad)

    def obtener_vecinos(self):
        """
        Genera los estados vecinos moviendo la casilla vacía.
        """
        vecinos = []
        x, y = self.posicion_vacia
        movimientos = {"Arriba": (x - 1, y), "Abajo": (x + 1, y), "Izquierda": (x, y - 1), "Derecha": (x, y + 1)}
        
        for mov, (nx, ny) in movimientos.items():
            if 0 <= nx < 3 and 0 <= ny < 3:
                nuevo_tablero = self.tablero.copy()
                nuevo_tablero[x, y], nuevo_tablero[nx, ny] = nuevo_tablero[nx, ny], nuevo_tablero[x, y]
                vecinos.append(puzzle_8(nuevo_tablero, self, mov, self.profundidad + 1, self.heuristica(nuevo_tablero)))
        
        return vecinos

    def heuristica(self, tablero):
        """
        Calcula la heurística de Manhattan: la suma de las distancias de cada número a su posición objetivo.
        """
        objetivo = {num: (i, j) for i, fila in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 0]]) for j, num in enumerate(fila)}
        return sum(abs(x - objetivo[num][0]) + abs(y - objetivo[num][1]) for x, fila in enumerate(tablero) for y, num in enumerate(fila) if num)

    def obtener_camino_tableros(self):
        """
        Retorna la secuencia de tableros desde el estado inicial hasta la solución.
        """
        camino, nodo = [], self
        while nodo:
            camino.append(nodo.tablero)
            nodo = nodo.padre
        return camino[::-1]

def resolver_8_puzzle(tablero_inicial):
    """
    Resuelve el 8-Puzzle utilizando el algoritmo A*.
    """
    nodo_inicial = puzzle_8(tablero_inicial, costo=0)
    cola_prioridad = [nodo_inicial]
    visitados = set()
    
    while cola_prioridad:
        actual = heapq.heappop(cola_prioridad)
        
        # Verifica si se ha alcanzado el estado objetivo
        if np.array_equal(actual.tablero, [[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
            return actual.obtener_camino_tableros()
        
        visitados.add(tuple(map(tuple, actual.tablero)))
        
        # Explora los vecinos y los agrega a la cola de prioridad si no han sido visitados
        for vecino in actual.obtener_vecinos():
            if tuple(map(tuple, vecino.tablero)) not in visitados:
                heapq.heappush(cola_prioridad, vecino)
    
    return None

# Ejemplo de uso
tablero_inicial = [[8, 5, 7], [6, 1, 4], [2, 3, 0]]
solucion = resolver_8_puzzle(tablero_inicial)

if solucion:
    print("Secuencia de tableros hasta la solución:")
    for paso, tablero in enumerate(solucion):
        print(f"Paso {paso}:")
        print(tablero, "\n")
else:
    print("No se encontró una solución.")
    
    
    