class ConnectionError(Exception):
    def __init__(self):
        super(ConnectionError, self).__init__("Can't connect to remote API "
                                              "server")


class SceneLoadingError(Exception):
    def __init__(self, path, code=None):
        super(SceneLoadingError, self).__init__("Error loading scene {}. Code {}"
                                                .format(path, code))


class WalkerLoadingError(Exception):
    def __init__(self):
        super(WalkerLoadingError, self).__init__("Error loading walker")


class MotorLoadingError(Exception):
    def __init__(self):
        super(MotorLoadingError, self).__init__("Error getting motor position")


class HandleLoadingError(Exception):
    def __init__(self, handle):
        super(HandleLoadingError, self).__init__("Error loading handle {}"
                                                 .format(handle))


class WalkerSaveError(Exception):
    def __init__(self, filename):
        super(WalkerSaveError, self).__init__("Error saving walker to file {}"
                                              .format(filename))


class RemoveWalkerError(Exception):
    def __init__(self):
        super(RemoveWalkerError, self).__init__("Error removing walker")
