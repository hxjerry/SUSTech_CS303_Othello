#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mcts.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mcts, m) {
    m.doc() = "MCTS module for Othello/Reversi";

    py::class_<mcts>(m, "MCTS")
        .def(py::init<>())
        .def_readwrite("root_key", &mcts::root_key)

        // release GIL for potentially long-running operations
        .def("select", &mcts::select, py::call_guard<py::gil_scoped_release>(),
             "Perform a single selection/expansion step; returns bool")
        .def("batch_select", &mcts::batch_select, py::call_guard<py::gil_scoped_release>(),
             py::arg("max_pending"), py::arg("max_explore"),
             "Perform batch select; returns number of selections made")

        .def("get_pending_nodes", &mcts::get_pending_nodes, py::call_guard<py::gil_scoped_release>(),
             "Get list of pending node keys (vector of (player, opponent) pairs)")

        .def("apply_evaluation", &mcts::apply_evaluation, py::call_guard<py::gil_scoped_release>(),
             py::arg("values"), py::arg("policies"),
             "Apply network evaluations: values (vector<float>), policies (vector<array<float,64>>)")

        .def("rebuild_tree", &mcts::rebuild_tree, py::call_guard<py::gil_scoped_release>(),
             "Rebuild tree from root_key")
        .def("take_action", &mcts::take_action, py::call_guard<py::gil_scoped_release>(),
             py::arg("move_idx"), "Apply chosen action to tree")
        .def("terminal_state", &mcts::terminal_state, py::call_guard<py::gil_scoped_release>(),
             "Return whether current state is terminal")
        .def("get_best_action", &mcts::get_best_action, py::call_guard<py::gil_scoped_release>(),
             py::arg("temperature"), "Return best action index (u_int64_t)")
        .def("get_policy", &mcts::get_policy, py::call_guard<py::gil_scoped_release>(),
             "Return search policy as std::array<float,64>");
}