
Shader "Skybox/NebulaPlxus"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 

    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }


const float FLIGHT_SPEED = 6.0;

const float FIELD_OF_VIEW = 1.05;


float3 getRayDirection(float2 fragCoord, float3 cameraDirection) {
    float2 uv = fragCoord.xy; // / iResolution.xy;
  
    const float screenWidth = 1.0;
    float originToScreen = screenWidth / 2.0 / tan(FIELD_OF_VIEW / 2.0);
    
    float3 screenCenter = originToScreen * cameraDirection;
    float3 baseX = normalize(cross(screenCenter, float3(0, -1.0, 0)));
    float3 baseY = normalize(cross(screenCenter, baseX));
    
    return normalize(screenCenter + (uv.x - 0.5) * baseX + (uv.y - 0.5)); // * iResolution.y / iResolution.x * baseY);
}

float4 getNebulaColor(float3 globalPosition, float3 rayDirection) {
    float3 color = float3(0.0,0,0);
    float spaceLeft = 1.0;
    
    const float layerDistance = 10.0;
    float rayLayerStep = rayDirection.z / layerDistance;
    
    const int steps = 4;
    for (int i = 0; i <= steps; i++) {
      float3 noiseeval = globalPosition + rayDirection * ((1.0 - frac(globalPosition.z / layerDistance) + float(i)) * layerDistance / rayDirection.z);
      noiseeval.xy += noiseeval.z;
        
        
        float value = 0.06 * tex2D(_MainTex, frac(noiseeval.xy / 60.0)).r;
         
        if (i == 0) {
            value *= 1.0 - frac(globalPosition.z / layerDistance);
        } else if (i == steps) {
            value *= frac(globalPosition.z / layerDistance);
        }
                
        color += spaceLeft * 2. * float3(value, value, value) * float3(.5, .3, 0.1);
        spaceLeft = max(0.0, spaceLeft - value * 2.0);
    }
    return float4(color, 1.0);
}

#define S(a, b, t) smoothstep(a, b, t)

float distLine(float2 p, float2 a, float2 b) {
    float2 pa = p - a;
    float2 ba = b - a;
    float t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba*t);
}

float lineL(float2 p, float2 a, float2 b) {
  float d = distLine(p, a, b);
    float m = S(0.03, 0.01, d);
    float d2 =  length(a - b);
    m *= S(1.2, 0.8, d2) * 0.5 + S(0.05, 0.03, abs(d2 - 0.75));
    return m;
}

float distTriangle(in float2 p, in float2 p0, in float2 p1, in float2 p2 )
{
  float2 e0 = p1 - p0;
  float2 e1 = p2 - p1;
  float2 e2 = p0 - p2;

  float2 v0 = p - p0;
  float2 v1 = p - p1;
  float2 v2 = p - p2;

  float2 pq0 = v0 - e0*clamp( dot(v0,e0)/dot(e0,e0), 0.0, 1.0 );
  float2 pq1 = v1 - e1*clamp( dot(v1,e1)/dot(e1,e1), 0.0, 1.0 );
  float2 pq2 = v2 - e2*clamp( dot(v2,e2)/dot(e2,e2), 0.0, 1.0 );
    
    float s = sign( e0.x*e2.y - e0.y*e2.x );
    float2 d = min( min( float2( dot( pq0, pq0 ), s*(v0.x*e0.y-v0.y*e0.x) ),
                       float2( dot( pq1, pq1 ), s*(v1.x*e1.y-v1.y*e1.x) )),
                       float2( dot( pq2, pq2 ), s*(v2.x*e2.y-v2.y*e2.x) ));

  return -sqrt(d.x)*sign(d.y);
}

float triangleS(float2 p, float2 a, float2 b, float2 c){
  float d = distTriangle(p, a, b, c);
    float m = S(0.03, 0.01, d);
    float d2 =  length(a - b);
    m *= S(1.2, 0.8, d2) * 0.5 + S(0.05, 0.03, abs(d2 - 0.75));
    return m;
}

float N21(float2 p){
  p = frac(p * float2(233.34, 851.73));
    p += dot(p, p + 23.45);
    return frac(p.x * p.y);
}

float2 N22(float2 p){
  float n = N21(p);
    return float2(n, N21(p + n));
}

float2 getPos(float2 id, float2 offset){
    float2 n = N22(id + offset) * _Time.y;
    return offset + sin(n) * 0.4;
}

float layer(float2 uv){
  float2 gv = frac(uv) - 0.5;
    float2 id = floor(uv);
    float2 p[9];
    int i = 0;
    for(float y = -1.0; y <= 1.0; y++){
      for(float x = -1.0; x <= 1.0; x++){
          p[i++] = getPos(id, float2(x, y));
      }    
    }
    
    
    float t = _Time.y * 10.0;
    float m = 0.0;
    for(int i = 0; i < 9; i++){
      m += lineL(gv, p[4], p[i]);
        
        float2 j = (p[i] - gv) * 20.0;
        float sparkle = 1.0 / dot(j, j);
        
        m += sparkle * (sin(t + frac(p[i].x) * 10.0) * 0.5 + 0.5);
        
        for(int yi= i + 1; yi < 9; yi++){
        for(int zi= yi + 1; zi < 9; zi++){
                
                float len1 = abs(length(p[i] - p[yi]));
                float len2 = abs(length(p[yi] - p[zi]));
                float len3 = abs(length(p[i] - p[zi]));
                
                if((len1 + len2 + len3) < 2.8){
                  m += triangleS(gv, p[i], p[yi], p[zi]) * 0.8;
                }
        }
      }
    }
    m += lineL(gv, p[1], p[3]);
    m += lineL(gv, p[1], p[5]);
    m += lineL(gv, p[7], p[3]);
    m += lineL(gv, p[7], p[5]);

    return m;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin


    //nebula
    float3 movementDirection = normalize(float3(0.01, 0.0, 1.0));
    
    float3 rayDirection = getRayDirection(fragCoord, movementDirection);
    
    float3 globalPosition = float3(3.14159, 3.14159, 0.0) + (_Time.y + 1000.0) * FLIGHT_SPEED * movementDirection;
    
    fragColor = getNebulaColor(globalPosition, rayDirection);
    
    // plexus
    float2 uv = v.uv; //(fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    float m = 0.0;
    float t = _Time.y * 0.1;
        
    for(float i = 0.0; i < 1.0; i += 1.0 / 4.0){
        float z = frac(i + t);
        float size = lerp(10.0, 0.5, z);
        float fade = S(0.0, 0.1, z) * S(1.0, 0.8, z);
        
        m += layer(uv * size + i * 20.0) * fade;
    }
    
    
    float3 base = float3(0.5, 0.3, 0.1);
    float3 col = m * base * 0.1;
    
    col -= uv.y * 0.5 * base;
        
    fragColor += float4(col,1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}

