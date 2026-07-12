import numpy as np
import forge


class TestLogger:
    def test_set_level(self):
        forge.Logger.set_level(forge.LogLevel.DEBUG)
        assert forge.Logger.level() == forge.LogLevel.DEBUG.value
        forge.Logger.set_level(forge.LogLevel.WARN)

    def test_python_sink(self, loaded_context):
        logs = []

        def capture(level, msg):
            logs.append((level, msg))

        forge.Logger.set_level(forge.LogLevel.WARN)
        forge.Logger.set_python_sink(capture)

        ids = np.array([1], dtype=np.int32)
        loaded_context.reset_kv()
        loaded_context.forward(ids)

        forge.Logger.reset_sink()
        forge.Logger.set_level(forge.LogLevel.WARN)

    def test_log_level_filtering(self, loaded_context):
        logs = []

        def capture(level, msg):
            logs.append((level, msg))

        forge.Logger.set_level(forge.LogLevel.LOG_ERROR)
        forge.Logger.set_python_sink(capture)

        ids = np.array([1], dtype=np.int32)
        loaded_context.forward(ids)

        for lvl, _ in logs:
            assert lvl <= forge.LogLevel.LOG_ERROR.value

        forge.Logger.reset_sink()
        forge.Logger.set_level(forge.LogLevel.WARN)

    def test_log_level_enum(self):
        assert forge.LogLevel.NONE.value == 0
        assert forge.LogLevel.LOG_ERROR.value == 1
        assert forge.LogLevel.WARN.value == 2
        assert forge.LogLevel.INFO.value == 3
        assert forge.LogLevel.DEBUG.value == 4
        assert forge.LogLevel.TRACE.value == 5
