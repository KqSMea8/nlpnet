import os
from ctypes import c_int, cdll, POINTER, byref
cur_path=os.path.dirname('./')
so_path=os.path.join(cur_path,'pointer.so')
so_path=os.path.normpath(so_path)
pointer=cdll.LoadLibrary(so_path)
a, b, r=c_int(10), c_int(20),c_int(0)
pointer.add(byref(a),byref(b),byref(r))
print r.value
