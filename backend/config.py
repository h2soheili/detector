from typing import List, Optional

from pydantic_settings import BaseSettings

from backend.global_log import structlog

logger = structlog.get_logger(__name__)


# pwd = os.getcwd()
# is_colab = pwd == '/content'
# p = pwd.replace('ai', '').replace('csv_files', '').strip() if not is_colab else "/content/algo_runner"
# env_file = os.path.join(p, '.env')
# # env_file = "/var/www/.env
# print('Config env_file_path is:', env_file)
# print("os.getcwd() - >>>> ", os.getcwd())


class Settings(BaseSettings):
    SECRET_KEY: str = ""
    # 60 minutes * 24 hours * x days = x days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30
    SERVER_NAME: Optional[str] = None
    BACKEND_CORS_ORIGINS: List[str] = []
    REDIS_DSN: Optional[str] = None

    OBJECT_STORAGE_URL: Optional[str] = ''
    OBJECT_STORAGE_ACCESS_KEY: Optional[str] = ''
    OBJECT_STORAGE_SECRET_KEY: Optional[str] = ''

    SHOW_LOG: Optional[bool] = False

    SQL_DSN: Optional[str] = None

    # model_config = SettingsConfigDict(env_file=env_file, env_file_encoding='utf-8')


settings = Settings()
print(settings)
