from multiview_tasks.tasks import blind_insertion, color_button

# Task name -> {level: robosuite env name}
TASK_ENV_NAMES = {
    blind_insertion.BlindInsertionBottleCabinet.TASK_NAME: blind_insertion.ENV_NAMES,
    color_button.ColorButtonBox.TASK_NAME: color_button.ENV_NAMES,
}
