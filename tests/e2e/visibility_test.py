from __future__ import annotations

import pytest
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme.app import MainWindow

from ..conftest import close_or_pause


@pytest.mark.gui
def test_toggle_all_shapes(
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    label_list = annotated_win._docks.label_list

    assert len(canvas.shapes) == 5
    for shape in canvas.shapes:
        assert canvas.is_shape_visible(shape)

    annotated_win.toggle_shape_visibility(False)
    qtbot.wait(50)

    for item in label_list:
        assert item.checkState() == Qt.Unchecked
    for shape in canvas.shapes:
        assert not canvas.is_shape_visible(shape)

    annotated_win.toggle_shape_visibility(True)
    qtbot.wait(50)

    for item in label_list:
        assert item.checkState() == Qt.Checked
    for shape in canvas.shapes:
        assert canvas.is_shape_visible(shape)

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_toggle_individual_shape(
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    label_list = annotated_win._docks.label_list

    assert len(canvas.shapes) == 5

    first_item = label_list[0]
    first_shape = first_item.shape()
    assert first_shape is not None
    assert canvas.is_shape_visible(first_shape)

    first_item.setCheckState(Qt.Unchecked)
    qtbot.wait(50)
    assert not canvas.is_shape_visible(first_shape)

    first_item.setCheckState(Qt.Checked)
    qtbot.wait(50)
    assert canvas.is_shape_visible(first_shape)

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_visibility_preserved_when_undoing_unrelated_edit(
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    label_list = annotated_win._docks.label_list

    assert len(canvas.shapes) == 5

    hidden_index = 1
    label_list[hidden_index].setCheckState(Qt.Unchecked)
    qtbot.wait(50)
    assert not canvas.is_shape_visible(canvas.shapes[hidden_index])

    canvas.shapes[0].points[0] += QPointF(5.0, 5.0)
    canvas.backup_shapes()

    annotated_win.undo_shape_edit()
    qtbot.wait(50)

    assert len(canvas.shapes) == 5
    for i, shape in enumerate(canvas.shapes):
        expected_visible = i != hidden_index
        assert canvas.is_shape_visible(shape) is expected_visible
        expected_state = Qt.Checked if expected_visible else Qt.Unchecked
        assert label_list[i].checkState() == expected_state

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)


@pytest.mark.gui
def test_undo_recovers_accidental_hide(
    qtbot: QtBot,
    annotated_win: MainWindow,
    pause: bool,
) -> None:
    canvas = annotated_win._canvas_widgets.canvas
    label_list = annotated_win._docks.label_list

    assert len(canvas.shapes) == 5
    hidden_index = 1

    label_list[hidden_index].setCheckState(Qt.Unchecked)
    qtbot.wait(50)
    assert not canvas.is_shape_visible(canvas.shapes[hidden_index])
    assert annotated_win._actions.undo.isEnabled()

    annotated_win._actions.undo.trigger()
    qtbot.wait(50)

    assert len(canvas.shapes) == 5
    for i, shape in enumerate(canvas.shapes):
        assert canvas.is_shape_visible(shape)
        assert label_list[i].checkState() == Qt.Checked

    close_or_pause(qtbot=qtbot, widget=annotated_win, pause=pause)
