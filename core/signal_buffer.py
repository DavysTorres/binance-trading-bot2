#core/signal_buffer.py

import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

class EnhancedSignalBuffer:
    """
    Buffer inteligente de señales con:
    - Priorización dinámica
    - Gestión de ciclo de vida
    - Integración con análisis optimizado
    - Métricas de performance
    """
    
    def __init__(self, buffer_file: str = "signals_buffer.json", config: Dict = None):
        self.buffer_file = buffer_file
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuración
        self.max_signals = self.config.get('max_buffer_signals', 50)
        self.signal_ttl_minutes = self.config.get('signal_ttl_minutes', 30)
        self.cleanup_interval = self.config.get('cleanup_interval_minutes', 5)
        
        # Estadísticas
        self.stats = {
            'signals_added': 0,
            'signals_removed': 0,
            'signals_expired': 0,
            'last_cleanup': None
        }
        
        self._ensure_buffer_file()
    
    def _ensure_buffer_file(self):
        """Asegura que el archivo buffer exista"""
        try:
            if not os.path.exists(self.buffer_file):
                initial_data = {
                    "signals": [],
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "2.0",
                        "total_processed": 0
                    }
                }
                with open(self.buffer_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error creando archivo buffer: {e}")
    
    def add_signals(self, new_signals: List[Dict]):
        """
        Añade señales al buffer con metadatos mejorados
        Integrado con el análisis optimizado
        """
        try:
            buffer_data = self._load_buffer()
            existing_signals = buffer_data.get("signals", [])
            
            # Limpiar señales expiradas primero
            existing_signals = self._clean_expired_signals(existing_signals)
            
            # Procesar nuevas señales
            added_count = 0
            for signal in new_signals:
                if self._should_add_signal(signal, existing_signals):
                    enhanced_signal = self._enhance_signal_data(signal)
                    existing_signals.append(enhanced_signal)
                    added_count += 1
                    self.stats['signals_added'] += 1
            
            # Ordenar y limitar tamaño del buffer
            existing_signals.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
            existing_signals = existing_signals[:self.max_signals]
            
            # Actualizar metadata
            buffer_data["signals"] = existing_signals
            buffer_data["metadata"].update({
                "last_updated": datetime.now().isoformat(),
                "total_signals": len(existing_signals),
                "signals_added_this_cycle": added_count
            })
            
            # Guardar
            self._save_buffer(buffer_data)
            
            self.logger.info(f"📁 Buffer: +{added_count} señales, total: {len(existing_signals)}")
            
        except Exception as e:
            self.logger.error(f"Error añadiendo señales al buffer: {e}")
    
    def _should_add_signal(self, signal: Dict, existing_signals: List[Dict]) -> bool:
        """Determina si una señal debe ser añadida al buffer"""
        # Verificar score mínimo
        min_score = self.config.get('min_buffer_score', 60)
        signal_score = signal.get('composite_score', 0)
        if signal_score < min_score:
            return False
        
        # Verificar si ya existe una señal similar
        symbol = signal.get('symbol')
        signal_type = signal.get('type')
        
        for existing in existing_signals:
            if (existing.get('symbol') == symbol and 
                existing.get('type') == signal_type and
                self._are_signals_similar(signal, existing)):
                return False
        
        return True
    
    def _are_signals_similar(self, signal1: Dict, signal2: Dict) -> bool:
        """Determina si dos señales son similares"""
        try:
            # Mismo símbolo y tipo
            if (signal1.get('symbol') != signal2.get('symbol') or
                signal1.get('type') != signal2.get('type')):
                return False
            
            # Score similar (dentro de 10 puntos)
            score_diff = abs(signal1.get('composite_score', 0) - signal2.get('composite_score', 0))
            if score_diff > 10:
                return False
            
            # Timestamp similar (dentro de 15 minutos)
            time1 = datetime.fromisoformat(signal1.get('timestamp', datetime.now().isoformat()))
            time2 = datetime.fromisoformat(signal2.get('timestamp', datetime.now().isoformat()))
            time_diff = abs((time1 - time2).total_seconds() / 60)
            
            return time_diff <= 15
            
        except:
            return False
    
    def _enhance_signal_data(self, signal: Dict) -> Dict:
        """Añade metadatos adicionales a la señal"""
        enhanced = signal.copy()
        
        # Calcular score de prioridad
        enhanced['priority_score'] = self._calculate_priority_score(signal)
        
        # Metadatos de buffer
        enhanced['buffer_data'] = {
            'added_at': datetime.now().isoformat(),
            'last_checked': datetime.now().isoformat(),
            'check_count': 0,
            'status': 'PENDING'
        }
        
        # Información de timeframes
        if 'timeframe_signals' in enhanced:
            timeframes = list(enhanced['timeframe_signals'].keys())
            enhanced['timeframe_count'] = len(timeframes)
            enhanced['primary_timeframe'] = timeframes[0] if timeframes else None
        
        # Categorización
        enhanced['category'] = self._categorize_signal(signal)
        
        return enhanced
    
    def _calculate_priority_score(self, signal: Dict) -> float:
        """Calcula score de prioridad para el buffer"""
        base_score = signal.get('composite_score', 0)
        
        # Bonus por validación histórica
        historical_validation = signal.get('historical_validation', {})
        if historical_validation.get('historical_confidence', 0.5) > 0.7:
            base_score *= 1.1
        
        # Bonus por multi-timeframe
        timeframe_count = len(signal.get('timeframe_signals', {}))
        if timeframe_count >= 3:
            base_score *= 1.05
        
        # Bonus por score ajustado
        adjusted_score = signal.get('adjusted_score', base_score)
        if adjusted_score > base_score:
            base_score = adjusted_score
        
        return min(base_score, 100)
    
    def _categorize_signal(self, signal: Dict) -> str:
        """Categoriza la señal para mejor organización"""
        score = signal.get('composite_score', 0)
        
        if score >= 80:
            return "HIGH_PRIORITY"
        elif score >= 70:
            return "MEDIUM_PRIORITY"
        elif score >= 60:
            return "LOW_PRIORITY"
        else:
            return "MONITOR_ONLY"
    
    def get_signals_for_reevaluation(self, max_signals: int = 10) -> List[Dict]:
        """
        Obtiene señales listas para re-evaluación
        Considera prioridad y tiempo desde última verificación
        """
        try:
            buffer_data = self._load_buffer()
            signals = buffer_data.get("signals", [])
            
            # Filtrar señales válidas
            valid_signals = []
            for signal in signals:
                if self._is_signal_ready_for_reevaluation(signal):
                    valid_signals.append(signal)
            
            # Ordenar por prioridad
            valid_signals.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
            
            # Actualizar contador de verificaciones
            for signal in valid_signals[:max_signals]:
                signal['buffer_data']['last_checked'] = datetime.now().isoformat()
                signal['buffer_data']['check_count'] += 1
            
            # Guardar cambios
            self._save_buffer(buffer_data)
            
            return valid_signals[:max_signals]
            
        except Exception as e:
            self.logger.error(f"Error obteniendo señales para re-evaluación: {e}")
            return []
    
    def _is_signal_ready_for_reevaluation(self, signal: Dict) -> bool:
        """Verifica si una señal está lista para re-evaluación"""
        try:
            buffer_data = signal.get('buffer_data', {})
            last_checked = datetime.fromisoformat(buffer_data.get('last_checked', datetime.now().isoformat()))
            
            # Tiempo mínimo entre verificaciones (2 minutos)
            min_interval = timedelta(minutes=2)
            time_since_last_check = datetime.now() - last_checked
            
            return (time_since_last_check >= min_interval and 
                    not self._is_signal_expired(signal))
                    
        except:
            return False
    
    def _clean_expired_signals(self, signals: List[Dict]) -> List[Dict]:
        """Limpia señales expiradas del buffer"""
        valid_signals = []
        expired_count = 0
        
        for signal in signals:
            if not self._is_signal_expired(signal):
                valid_signals.append(signal)
            else:
                expired_count += 1
                self.stats['signals_expired'] += 1
        
        if expired_count > 0:
            self.logger.info(f"🗑️ {expired_count} señales expiradas removidas")
        
        return valid_signals
    
    def _is_signal_expired(self, signal: Dict) -> bool:
        """Verifica si una señal ha expirado"""
        try:
            signal_time = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
            expiration_time = signal_time + timedelta(minutes=self.signal_ttl_minutes)
            return datetime.now() > expiration_time
        except:
            return True
    
    def update_signal_status(self, symbol: str, signal_type: str, status: str, metadata: Dict = None):
        """Actualiza el estado de una señal específica"""
        try:
            buffer_data = self._load_buffer()
            signals = buffer_data.get("signals", [])
            
            for signal in signals:
                if (signal.get('symbol') == symbol and 
                    signal.get('type') == signal_type):
                    
                    signal['buffer_data']['status'] = status
                    signal['buffer_data']['updated_at'] = datetime.now().isoformat()
                    
                    if metadata:
                        signal['buffer_data'].update(metadata)
                    
                    break
            
            self._save_buffer(buffer_data)
            self.logger.info(f"🔄 Señal {symbol} {signal_type} actualizada a {status}")
            
        except Exception as e:
            self.logger.error(f"Error actualizando estado de señal: {e}")
    
    def remove_signal(self, symbol: str, signal_type: str):
        """Remueve una señal específica del buffer"""
        try:
            buffer_data = self._load_buffer()
            signals = buffer_data.get("signals", [])
            
            updated_signals = [
                signal for signal in signals 
                if not (signal.get('symbol') == symbol and signal.get('type') == signal_type)
            ]
            
            removed_count = len(signals) - len(updated_signals)
            buffer_data["signals"] = updated_signals
            
            self._save_buffer(buffer_data)
            self.stats['signals_removed'] += removed_count
            
            if removed_count > 0:
                self.logger.info(f"🗑️ Señal removida: {symbol} {signal_type}")
                
        except Exception as e:
            self.logger.error(f"Error removiendo señal: {e}")
    
    def get_buffer_stats(self) -> Dict:
        """Obtiene estadísticas detalladas del buffer"""
        try:
            buffer_data = self._load_buffer()
            signals = buffer_data.get("signals", [])
            
            if not signals:
                return {
                    "total": 0,
                    "by_priority": {},
                    "by_status": {},
                    "avg_score": 0
                }
            
            # Estadísticas por prioridad
            priority_counts = {}
            status_counts = {}
            scores = []
            
            for signal in signals:
                category = signal.get('category', 'UNKNOWN')
                status = signal.get('buffer_data', {}).get('status', 'UNKNOWN')
                score = signal.get('composite_score', 0)
                
                priority_counts[category] = priority_counts.get(category, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1
                scores.append(score)
            
            return {
                "total": len(signals),
                "by_priority": priority_counts,
                "by_status": status_counts,
                "avg_score": round(sum(scores) / len(scores), 2),
                "min_score": min(scores),
                "max_score": max(scores),
                "performance_stats": self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo stats del buffer: {e}")
            return {"total": 0, "error": str(e)}
    
    def _load_buffer(self) -> Dict:
        """Carga el buffer desde archivo"""
        try:
            with open(self.buffer_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error cargando buffer: {e}")
            return {"signals": [], "metadata": {}}
    
    def _save_buffer(self, buffer_data: Dict):
        """Guarda el buffer en archivo"""
        try:
            with open(self.buffer_file, 'w') as f:
                json.dump(buffer_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error guardando buffer: {e}")
    
    def perform_maintenance(self):
        """Realiza mantenimiento periódico del buffer"""
        try:
            buffer_data = self._load_buffer()
            signals = buffer_data.get("signals", [])
            
            # Limpiar expiradas
            initial_count = len(signals)
            signals = self._clean_expired_signals(signals)
            
            # Recalcular prioridades
            for signal in signals:
                signal['priority_score'] = self._calculate_priority_score(signal)
                signal['category'] = self._categorize_signal(signal)
            
            buffer_data["signals"] = signals
            self._save_buffer(buffer_data)
            
            self.stats['last_cleanup'] = datetime.now().isoformat()
            
            cleaned_count = initial_count - len(signals)
            if cleaned_count > 0:
                self.logger.info(f"🧹 Maintenance: {cleaned_count} señales limpiadas")
                
        except Exception as e:
            self.logger.error(f"Error en mantenimiento del buffer: {e}")