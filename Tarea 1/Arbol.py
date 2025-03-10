from Nodo import Nodo
class Arbol:
   
    def __init__(self, dato):
        self.raiz = Nodo(dato)
    
    def __init__(self):
        self.raiz = None

    def __insertar_recursivo(self, nodo, dato):
        if self.raiz is None:
                self.raiz = Nodo(dato)
        else:
            if dato < nodo.dato:
                if nodo.izquierda is None:
                    nodo.izquierda = Nodo(dato)
                else:
                    self.__insertar_recursivo(nodo.izquierda, dato)
            else:
                if nodo.derecha is None:
                    nodo.derecha = Nodo(dato)
                else:
                    self.__insertar_recursivo(nodo.derecha, dato)

    def __inorden_recursivo(self, nodo):
        if nodo is not None:
            self.__inorden_recursivo(nodo.izquierda)
            print(nodo.dato, end=", ")
            self.__inorden_recursivo(nodo.derecha)

    def __preorden_recursivo(self, nodo):
        if nodo is not None:
            print(nodo.dato, end=", ")
            self.__preorden_recursivo(nodo.izquierda)
            self.__preorden_recursivo(nodo.derecha)

    def __postorden_recursivo(self, nodo):
        if nodo is not None:
            self.__postorden_recursivo(nodo.izquierda)
            self.__postorden_recursivo(nodo.derecha)
            print(nodo.dato, end=", ")

    def __buscar(self, nodo, busqueda):
        if nodo is None:
            return None
        if nodo.dato == busqueda:
            return nodo
        if busqueda < nodo.dato:
            return self.__buscar(nodo.izquierda, busqueda)
        else:
            return self.__buscar(nodo.derecha, busqueda)


    def insertar(self, dato):
        self.__insertar_recursivo(self.raiz, dato)

    def inorden(self):
        print("Imprimiendo árbol inorden: ")
        self.__inorden_recursivo(self.raiz)
        print("")

    def preorden(self):
        print("Imprimiendo árbol preorden: ")
        self.__preorden_recursivo(self.raiz)
        print("")

    def postorden(self):
        print("Imprimiendo árbol postorden: ")
        self.__postorden_recursivo(self.raiz)
        print("")

    def buscar(self, busqueda):
        return self.__buscar(self.raiz, busqueda)