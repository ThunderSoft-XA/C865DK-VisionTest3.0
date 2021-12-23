/**
 */
package com.thundercomm.gateway.data;



import java.io.Serializable;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.thingsboard.server.common.data.kv.KvEntry;
import org.thingsboard.server.common.data.kv.TsKvEntry;

/**
 */
@Data
@AllArgsConstructor
@RequiredArgsConstructor
@Builder
/**
 * 设备实例化的类，用于内存中设备信息的映射 使用Serizlizable用来做持久化
 * @see Serizlizable
 *  @Describe
 */
public class DeviceData implements Serializable {

    private final String name;
    private final String type;
    private final List<KvEntry> attributes;
    private final List<TsKvEntry> telemetry;
    private int timeout;

    @Override
    public String toString() {
        return "DeviceData{" +
                "name='" + name + '\'' +
                ", type='" + type + '\'' +
                ", attributes=" + attributes +
                ", telemetry=" + telemetry +
                ", timeout=" + timeout +
                '}';
    }

}
