
package com.thundercomm.rtsp;

/**
 * Video stream packet data.
 *
 *
 */
public class TsPcmData {
    /**
     * Video stream packet.
     */
    private byte[] pcm;

    /**
     * Packet size.
     */
    private int size;

    private int nalutype;

    /**
     * TSPcmData construction.
     * 
     * @param pcm Video stream packet
     * @param size Packet size
     */
    public TsPcmData(byte[] pcm, int size) {
        super();
        this.pcm = new byte[size];
        this.pcm = pcm;
        this.size = size;
    }

    /**
     * get video stream packet.
     * 
     * @return video stream packet
     */
    public byte[] getPcm() {
        return pcm;
    }

    /**
     * set video stream packet.
     * 
     * @param pcm video stream packet
     */
    public void setPcm(byte[] pcm) {
        this.pcm = pcm;
    }

    /**
     * get packet size.
     * 
     * @return packet size
     */
    public int getSize() {
        return size;
    }

    /**
     * set packet size.
     * 
     * @param size packet size
     */
    public void setSize(int size) {
        this.size = size;
    }

    public int nalu_type;

}
