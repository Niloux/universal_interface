from utils.config import Config


def main():
    config = Config()
    print(config.config)
    print(config.get("camera"))


if __name__ == "__main__":
    main()
