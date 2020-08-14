import santa_chromo
import test_class
import warnings
warnings.filterwarnings("ignore")
"""
A = test_class.A()
print(__name__)
if __name__ == '__main__':
    print("start")
    A.mul_foo(10)
"""

santa = santa_chromo.SantaChromo(1)

while True:
    santa.optimize_all(10)
    print("10 step")
