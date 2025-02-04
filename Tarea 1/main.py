from Arbol import Arbol

arbol = Arbol()
arbol.insertar("Luis")
arbol.insertar("María José")
arbol.insertar("Maggie")
arbol.insertar("Leon")
arbol.insertar("Cuphead")
arbol.insertar("Aloy")
arbol.insertar("Jack")
nombre = input("Ingresa algo para agregar al árbol: ")
arbol.insertar(nombre)
arbol.preorden()
arbol.inorden()
arbol.postorden()

busqueda = input("Busca algo en el árbol: ")
nodo = arbol.buscar(busqueda)
if nodo is None:
    print(f"{busqueda} no existe")
else:
    print(f"{busqueda} sí existe")