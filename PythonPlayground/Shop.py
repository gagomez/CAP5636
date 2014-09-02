from collections import Iterable

__author__ = 'G'


class Shop:
    shopId = 0

    def __init__(self, name):
        self.name = name
        self.id = Shop.shopId
        self.itemPrices = {}
        Shop.shopId += 1

    def add_item(self, name, price):
        self.itemPrices[name] = price

    def remove_item(self, name):
        del self.itemPrices[name]

    def get_item_price(self, name):
        return self.itemPrices[name]

    def get_items(self):
        return self.itemPrices.keys()


s = Shop('my shop')
t = Shop('their shop')

print s.id
print t.id

s.add_item('apple', 2.01)
s.add_item('pear', 5.55)
s.add_item('strawberry', 1.44)

t.add_item('top round', 7.57)
t.add_item('ground beef', 5.43)


def extend(x, y):
    if y is Iterable:
        return x + y
    else:
        x.append(y)
        return x


def flatten(iterable):
    return reduce(extend, iterable, [])

items = flatten(map(lambda shop: shop.get_items(), [s, t]))

print items