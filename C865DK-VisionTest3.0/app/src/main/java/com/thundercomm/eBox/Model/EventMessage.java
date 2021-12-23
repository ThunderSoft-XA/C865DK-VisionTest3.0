package com.thundercomm.eBox.Model;

public class EventMessage {
    private String messageType;
    private int CameraId;

    public EventMessage(String messageType, int cameraId, Object object) {
        this.messageType = messageType;
        CameraId = cameraId;
        this.object = object;
    }

    public int getCameraId() {
        return CameraId;
    }

    public void setCameraId(int cameraId) {
        CameraId = cameraId;
    }

    private Object object;

    public EventMessage(String messageType, Object object) {
        this.messageType = messageType;
        this.object = object;
    }

    public String getMessageType() {
        return messageType;
    }

    public void setMessageType(String messageType) {
        this.messageType = messageType;
    }

    public Object getObject() {
        return object;
    }

    public void setObject(Object object) {
        this.object = object;
    }

    @Override
    public String toString() {
        return "EventMessage{" +
                "messageType='" + messageType + '\'' +
                ", CameraId=" + CameraId +
                ", object=" + object +
                '}';
    }
}
