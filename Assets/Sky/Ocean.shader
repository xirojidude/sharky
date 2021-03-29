
Shader "Skybox/Ocean"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunPos ("Sun Position", Color) = (1.0,1,1,1) 

        
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
            float4 _SunPos;

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


//afl_ext 2017-2019

#define DRAG_MULT 0.048
#define ITERATIONS_RAYMARCH 13
#define ITERATIONS_NORMAL 48

#define Mouse (iMouse.xy / iResolution.xy)
#define Resolution (iResolution.xy)
#define Time (_Time.y)

float2 wavedx(float2 position, float2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return float2(wave, -dx);
}

float getwaves(float2 position, int iterations){
    float iter = 0.0;
    float phase = 6.0;
    float speed = .3;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        float2 p = float2(sin(iter), cos(iter));
        float2 res = wavedx(position, p, speed, phase, Time);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = lerp(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws;
}

float raymarchwater(float3 camera, float3 start, float3 end, float depth){
    float3 pos = start;
    float h = 0.0;
    float hupper = depth;
    float hlower = 0.0;
    float2 zer = float2(0.0,0);
    float3 dir = normalize(end - start);
    for(int i=0;i<318;i++){
        h = getwaves(pos.xz * 0.1, ITERATIONS_RAYMARCH) * depth - depth;
        if(h + 0.01 > pos.y) {
            return distance(pos, camera);
        }
        pos += dir * (pos.y - h);
    }
    return -1.0;
}

float H = 0.0;
float3 normal(float2 pos, float e, float depth){
    float2 ex = float2(e, 0);
    H = getwaves(pos.xy * 0.1, ITERATIONS_NORMAL) * depth;
    float3 a = float3(pos.x, H, pos.y);
    return normalize(cross(normalize(a-float3(pos.x - e, getwaves(pos.xy * 0.1 - ex.xy * 0.1, ITERATIONS_NORMAL) * depth, pos.y)), 
                           normalize(a-float3(pos.x, getwaves(pos.xy * 0.1 + ex.yx * 0.1, ITERATIONS_NORMAL) * depth, pos.y + e))));
}
float3x3 rotmat(float3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return float3x3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s, 
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s, 
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

//float3 getRay(float2 uv){
//    uv = (uv * 2.0 - 1.0) * float2(1.0, 1.0);
//    float3 proj = normalize(float3(uv.x, uv.y, 1.0) + float3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);    
//    if(800. < 400.0) return proj;
//    float3 ray = float3(1,1,1);//rotmat(float3(0.0, -1.0, 0.0), 3.0 * (1.0 * 2.0 - 1.0)) * rotmat(float3(1.0, 0.0, 0.0), 1.5 * (1.0 * 2.0 - 1.0)) * proj;
//    return ray;
//}

float intersectPlane(float3 origin, float3 direction, float3 points, float3 normal)
{ 
    return clamp(dot(points - origin, normal) / dot(direction, normal), -1.0, 9991999.0); 
}

float3 extra_cheap_atmosphere(float3 raydir, float3 sundir){
    sundir.y = max(sundir.y, -0.07);
    float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
    float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
    float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
    float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
    float mymie = sundt * special_trick * 0.2;
    float3 suncolor = lerp(float3(1.0,1,1), max(float3(0.0,0,0), float3(1.0,1,1) - float3(5.5, 13.0, 22.4) / 22.4), special_trick2);
    float3 bluesky= float3(5.5, 13.0, 22.4) / 22.4 * suncolor;
    float3 bluesky2 = max(float3(0.0,0,0), bluesky - float3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
    bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
    return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0)) + mymie * suncolor;
} 
float3 getatm(float3 ray){
    return extra_cheap_atmosphere(ray, normalize(float3(1.0,1,1))) * 0.5;
    
}

float sun(float3 ray){
    float3 sd = normalize(float3(_SunPos.xyz*2.+float3(-1,-1,-1)));   
    return pow(max(0.0, dot(ray, sd)), 928.0) * 110.0;
}
float3 aces_tonemap(float3 color){  
    float3x3 m1 = float3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    float3x3 m2 = float3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );
    float3 v = mul(m1 , color);    
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(mul(m2 , (a / b)), 0.0, 1.0), float3(1.0 / 2.2,1.0 / 2.2,1.0 / 2.2)); 
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin


 //   vec2 uv = fragCoord.xy / iResolution.xy;
    
    float waterdepth = 2.1;
    float3 wfloor = float3(0.0, -waterdepth, 0.0);
    float3 wceil = float3(0.0, 0.0, 0.0);
    float3 orig = ro; //float3(0.0, 2.0, 0.0);
    float3 ray = rd;  //getRay(uv);
    float hihit = intersectPlane(orig, ray, wceil, float3(0.0, 1.0, 0.0));
    if(ray.y >= -0.01){
        float3 C = getatm(ray) * 2.0 + sun(ray);
        //tonemapping
        C = aces_tonemap(C);
        fragColor = float4( C,1.0);   
        return fragColor;
    }
    float lohit = intersectPlane(orig, ray, wfloor, float3(0.0, 1.0, 0.0));
    float3 hipos = orig + ray * hihit;
    float3 lopos = orig + ray * lohit;
    float dist = raymarchwater(orig, hipos, lopos, waterdepth);
    float3 pos = orig + ray * dist;

    float3 N = normal(pos.xz, 0.001, waterdepth);
    float2 velocity = N.xz * (1.0 - N.y);
    N = lerp(float3(0.0, 1.0, 0.0), N, 1.0 / (dist * dist * 0.01 + 1.0));
    float3 R = reflect(ray, N);
    float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));
    
    float3 C = fresnel * getatm(R) * 1.0 + fresnel * sun(R);
    //tonemapping
    C = aces_tonemap(C);
    
    fragColor = float4(C,1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}


