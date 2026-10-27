# utils/logger.py
"""
Sistema de logging centralizado para toda la aplicación.
Reemplaza los print() statements con logging profesional.
Compatible con Windows (maneja emojis correctamente).
"""
import logging
import sys
from pathlib import Path
from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Formatter que añade colores a los logs en consola."""
    
    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[33m',    # Amarillo
        'ERROR': '\033[31m',      # Rojo
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Añadir color al nivel
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler que maneja correctamente Unicode en Windows.
    Reemplaza emojis por texto ASCII si hay problemas de encoding.
    """
    
    # Mapeo de emojis a texto ASCII
    EMOJI_REPLACEMENTS = {
        '🤖': '[AI]',
        '📨': '[MSG]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARN]',
        '🔧': '[CONFIG]',
        '📋': '[STATE]',
        '🆔': '[ID]',
        '🎨': '[UI]',
        '🔄': '[SYNC]',
        '💬': '[CHAT]',
        '📍': '[MAP]',
        '⏳': '[PENDING]',
        '🚀': '[START]',
        '📊': '[DATA]',
        '📄': '[FILE]',
        '🗺️': '[PLAN]',
        '🏠': '[ROOM]',
        '🔗': '[LINK]',
        '🚪': '[DOOR]',
        '📐': '[DIM]',
        '💾': '[SAVE]',
        '🗂️': '[FOLDER]',
    }
    
    def emit(self, record):
        """
        Emite el log manejando errores de encoding.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Intentar escribir directamente
            try:
                stream.write(msg + self.terminator)
                stream.flush()
            except UnicodeEncodeError:
                # Si falla, reemplazar emojis por texto ASCII
                for emoji, replacement in self.EMOJI_REPLACEMENTS.items():
                    msg = msg.replace(emoji, replacement)
                
                # Intentar de nuevo con el texto limpio
                stream.write(msg + self.terminator)
                stream.flush()
                
        except Exception:
            self.handleError(record)


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configura y retorna un logger.
    
    Args:
        name: Nombre del logger (normalmente __name__ del módulo)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers si ya está configurado
    if logger.handlers:
        return logger
    
    # Configurar nivel según settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Handler para consola (con soporte Unicode mejorado)
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formato detallado en DEBUG, simple en otros niveles
    if settings.DEBUG_MODE:
        formatter = ColoredFormatter(
            '%(levelname)s | %(name)s:%(lineno)d | %(message)s'
        )
    else:
        formatter = ColoredFormatter(
            '%(levelname)s | %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional, solo en producción)
    if not settings.DEBUG_MODE:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / "app.log",
            encoding='utf-8'  # IMPORTANTE: UTF-8 para emojis en archivo
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Logger por defecto para imports rápidos
logger = setup_logger("app")