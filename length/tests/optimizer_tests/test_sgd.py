from length.optimizers.sgd import SGD
from length.tests import gradient_checker
from length.tests.utils import init


def test_sgd():
    learning_rate = 0.0017
    optimizer = SGD(learning_rate)

    gradients = init([
        [0.78266141, 0.87160521, 0.91545263, 0.41808932, 0.63775016],
        [0.16893565, 0.25077806, 0.88390805, 0.92372049, 0.0741453 ],
        [0.63734837, 0.28873811, 0.20229677, 0.12343409, 0.08427269]
    ])

    desired = init([
        [0.00133052, 0.00148173, 0.00155627, 0.00071075, 0.00108418],
        [0.00028719, 0.00042632, 0.00150264, 0.00157032, 0.00012605],
        [0.00108349, 0.00049085, 0.0003439 , 0.00020984, 0.00014326]
    ])

    deltas, = optimizer.run_update_rule((gradients,), None)
    gradient_checker.assert_allclose(deltas, desired)
