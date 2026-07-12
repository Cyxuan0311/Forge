import numpy as np
import pytest
import nanoinfer


class TestLogger:
    def test_set_level(self):
        nanoinfer.Logger.set_level(nanoinfer.LogLevel.DEBUG)
        assert nanoinfer.Logger.level() == nanoinfer.LogLevel.DEBUG.value
        nanoinfer.Logger.set_level(nanoinfer.LogLevel.WARN)

    def test_python_sink(self, loaded_context):
        logs = []
        def capture(level, msg):
            logs.append((level, msg))

        nanoinfer.Logger.set_level(nanoinfer.LogLevel.WARN)
        nanoinfer.Logger.set_python_sink(capture)

        ids = np.array([1], dtype=np.int32)
        loaded_context.reset_kv()
        loaded_context.forward(ids)

        nanoinfer.Logger.reset_sink()
        nanoinfer.Logger.set_level(nanoinfer.LogLevel.WARN)

    def test_log_level_filtering(self, loaded_context):
        logs = []
        def capture(level, msg):
            logs.append((level, msg))

        nanoinfer.Logger.set_level(nanoinfer.LogLevel.ERROR)
        nanoinfer.Logger.set_python_sink(capture)

        ids = np.array([1], dtype=np.int32)
        loaded_context.forward(ids)

        for lvl, _ in logs:
            assert lvl <= nanoinfer.LogLevel.ERROR.value

        nanoinfer.Logger.reset_sink()
        nanoinfer.Logger.set_level(nanoinfer.LogLevel.WARN)

    def test_log_level_enum(self):
        assert nanoinfer.LogLevel.NONE.value == 0
        assert nanoinfer.LogLevel.ERROR.value == 1
        assert nanoinfer.LogLevel.WARN.value == 2
        assert nanoinfer.LogLevel.INFO.value == 3
        assert nanoinfer.LogLevel.DEBUG.value == 4
        assert nanoinfer.LogLevel.TRACE.value == 5
