# ====================
# 듀얼 네트워크 (Plain CNN 버전)
# ====================

from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                         GlobalAveragePooling2D, Input, Dropout)
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import os

# 하이퍼파라미터 ------------------------------
DN_FILTERS = 256        # Conv 채널 수
DN_CONV_LAYERS = 4      # Conv 레이어 개수 (plain)
DN_INPUT_SHAPE = (6, 6, 2)
DN_OUTPUT_SIZE = 37     # 6*6 + pass
L2_REG = 5e-4
DROPOUT_RATE = 0.3
# ---------------------------------------------


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_model():
    inp = Input(shape=DN_INPUT_SHAPE)

    # ─────────── Conv 스택 (plain) ────────────
    x = inp
    for _ in range(DN_CONV_LAYERS):
        x = conv_block(x, DN_FILTERS)
    # ─────────────────────────────────────────

    # Global Average Pooling → FC 헤드
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

    # policy head
    p = Dense(DN_OUTPUT_SIZE, activation='softmax', name='pi',
              kernel_regularizer=l2(L2_REG))(x)

    # value head
    v = Dense(1, activation='tanh', name='v', kernel_regularizer=l2(L2_REG))(x)

    model = Model(inputs=inp, outputs=[p, v])
    return model


# 듀얼 네트워크 생성 및 저장 (최초 1회)

def dual_network():
    if os.path.exists('../model/best.h5'):
        return

    model = build_model()
    os.makedirs('../model/', exist_ok=True)
    model.save('../model/best.h5')

    K.clear_session()
    del model


if __name__ == '__main__':
    dual_network()
