from utils.config import Config


def main():
    config = Config()
    print(config.config)
    print(config.get("camera"))
    print(config.is_enable("ego_pose"))


if __name__ == "__main__":
    main()
